"""
Find a metaphor subspace with DAS, at the "." of the test sentence in 5-shot prompts
("The {noun} was {adj}. This sentence is:" -> " metaphor" / " literal").

  M1  pick the layer: activation patching at "." for every layer.
  M2  global DAS on all domains (k = 16 and 32, 3 seeds). saves the R matrix
  M3  DAS per domain, and the global R tested on each domain

Validation uses held-out adjectives (one per domain). Outputs go to results/:
the result JSON (best layer, best k, IIA with 95% CI) and the R matrices, which
the steering and generation scripts use. See jobs/run_das_final_<model>.job.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer


#helpers
def fix_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bootstrap_ci(values, n_boot: int = 1000, ci: float = 0.95):
    arr = np.array(values, dtype=float)
    mu = float(arr.mean())
    if len(arr) < 2:
        return mu, mu, mu
    rng = np.random.default_rng(42)
    boots = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boots, 100 * (1 + ci) / 2))
    return mu, lo, hi


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"Loaded {model_name}")
    return model, tokenizer


class RotationMatrix(nn.Module):
    def __init__(self, d_model: int, k: int) -> None:
        super().__init__()
        Q, _ = torch.linalg.qr(torch.randn(d_model, d_model))
        self.R = nn.Parameter(Q[:, :k].clone())

    def project(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.R @ self.R.T

    def orthogonalise(self) -> None:
        with torch.no_grad():
            Q, _ = torch.linalg.qr(self.R)
            self.R.copy_(Q)

# best layer function
def kneedle_layer(layer_iia: dict) -> int:
    layers = sorted(layer_iia.keys())
    iia = np.array([layer_iia[l] for l in layers], dtype=float)
    if iia.max() < 0.05:
        return layers[int(np.argmax(iia))]
    n = len(layers)
    x_n = np.arange(n, dtype=float) / max(n - 1, 1)
    y_r = iia.max() - iia.min()
    y_n = (iia - iia.min()) / (y_r if y_r > 0 else 1.0)
    knee_idx = int(np.argmax(y_n - x_n))
    return layers[knee_idx]


# configurations

MODEL_NAME        = os.environ.get("DAS_MODEL", "google/gemma-2-9b-it")
BATCH_SIZE        = int(os.environ.get("BATCH_SIZE",    "16"))
DAS_BATCH         = int(os.environ.get("DAS_BATCH",     "32"))
N_STEPS           = int(os.environ.get("DAS_N_STEPS",   "1000"))
K_VALUES          = [int(k) for k in os.environ.get("DAS_K_VALUES", "2,4,8,16").split(",")]
SEEDS             = [int(s) for s in os.environ.get("DAS_SEEDS",    "42,123,456").split(",")]
LAYER_START       = int(os.environ.get("LAYER_START",   "0"))
LAYER_END         = int(os.environ.get("LAYER_END",     "999"))
N_HOLDOUT_ADJS    = int(os.environ.get("N_HOLDOUT_ADJS",   "4"))
N_HOLDOUT_SPLITS  = int(os.environ.get("N_HOLDOUT_SPLITS", "3"))
N_SHOTS           = int(os.environ.get("N_SHOTS",       "5"))
# M2_RESULTS: path to existing global DAS JSON — if set, skip M1+M2 and run M3 only.
M2_RESULTS        = os.environ.get("M2_RESULTS",        "")
# FORCE_LAYER: if set, skip layer sweep and use this layer directly for DAS training.
FORCE_LAYER       = int(os.environ.get("FORCE_LAYER", "-1"))
DATA_PATH         = os.environ.get(
    "METAPHOR_DATA",
    "data/gut-2016-new.csv",
)
MODEL_TAG    = MODEL_NAME.replace("/", "_").replace("-", "_").lower()
RESULTS_FILE = os.environ.get(
    "METAPHOR_DAS_RESULTS",
    f"results/results_metaphor_das_final_{MODEL_TAG}.json",
)
# Period token is 4 positions before last token: '. This sentence is:'
# = ['.', ' This', ' sentence', ' is', ':']. offset of 4 from the colon (this we have verified).
PERIOD_OFFSET = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCE_FRAME = "The {noun} was {adj}."

#categories taken from original paper (https://aclanthology.org/P16-1018.pdf)
CMT_DOMAIN: dict[str, str] = {
    "warm":      "TEMPERATURE", "cold":      "TEMPERATURE",
    "icy":       "TEMPERATURE", "heated":    "TEMPERATURE",
    "bright":    "LIGHT",       "brilliant": "LIGHT",       "dim":  "LIGHT",
    "clear":     "CLARITY",     "clean":     "CLARITY",
    "rough":     "TEXTURE",     "smooth":    "TEXTURE",     "soft": "TEXTURE",
    "dense":     "SUBSTANCE",   "solid":     "SUBSTANCE",   "murky":"SUBSTANCE",
    "bitter":    "TASTE",       "sweet":     "TASTE",       "sour": "TASTE",
    "strong":    "STRENGTH",    "weak":      "STRENGTH",    "heavy":"STRENGTH",
    "deep":      "DEPTH",       "shallow":   "DEPTH",
}

#prompt 
def make_prompt(adj: str, noun: str, few_shot_prefix: str = "") -> str:
    """Plain Marks and Tegmark continuation prompt."""
    test_sent = SENTENCE_FRAME.format(noun=noun, adj=adj)
    return f"{few_shot_prefix}{test_sent} This sentence is:"

# token extraction
def get_met_lit_toks(tokenizer) -> tuple[int, int]:
    """Return (met_tok, lit_tok). Uses ' metaphor'/' literal' — single token for all models."""
    met = tokenizer(" metaphor", add_special_tokens=False).input_ids[0]
    lit = tokenizer(" literal",  add_special_tokens=False).input_ids[0]
    return met, lit

# build few shot
def build_few_shot_prefix(
    df: pd.DataFrame, exclude_adj: str, n_shots: int, rng: random.Random,
) -> str:
    """Sample n_shots real pairs (exclude_adj held out), balanced lit/met."""
    pool     = df[df["adj"] != exclude_adj]
    met_rows = pool[pool["label"] == 1].to_dict("records")
    lit_rows  = pool[pool["label"] == 0].to_dict("records")
    rng.shuffle(met_rows); rng.shuffle(lit_rows)
    n_met    = n_shots // 2; n_lit = n_shots - n_met
    examples = met_rows[:n_met] + lit_rows[:n_lit]
    rng.shuffle(examples)
    lines = []
    for ex in examples:
        sent  = SENTENCE_FRAME.format(noun=ex["noun"], adj=ex["adj"])
        label = "metaphor" if ex["label"] == 1 else "literal"
        lines.append(f"{sent} This sentence is: {label}.")
    return "\n".join(lines) + "\n"


# make data in the correct prompt format
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["adj"]    = df["expression"].str.strip()
    df["noun"]   = df["tenor"].str.strip()
    df["label"]  = (df["label"].str.strip() == "m").astype(int)
    df["domain"] = df["adj"].map(CMT_DOMAIN)
    # Build per-adjective few-shot prefix (test adj excluded from its own prefix)
    rng = random.Random(42)
    prefix_cache = {adj: build_few_shot_prefix(df, adj, N_SHOTS, rng)
                    for adj in df["adj"].unique()}
    df["prompt"] = [
        make_prompt(r.adj, r.noun, prefix_cache[r.adj])
        for r in df.itertuples()
    ]
    print(f"Loaded {len(df)} pairs  "
          f"({df['label'].sum()} metaphorical, {(df['label']==0).sum()} literal)")
    return df


#extraction of hidden states
def extract_hidden(model, tokenizer, prompts: list[str], layer: int) -> np.ndarray:
    """Extract residual stream at period token ('.' in test sentence) at given layer."""
    all_hidden = []
    captured   = {}

    def hook_fn(_module, _input, output): #during forward pass through a layer, the hook 'fires' and saves the output tensor
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().cpu().float() 

    hook = model.model.layers[layer].register_forward_hook(hook_fn) #setup hook to listen during forward pass saving the out tensors at a layer
    try:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch  = prompts[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(DEVICE)
            with torch.no_grad():
                model(**inputs)
            hidden     = captured["h"]
            period_pos = inputs["attention_mask"].cpu().sum(dim=1) - 1 - PERIOD_OFFSET
            for j in range(len(batch)):
                all_hidden.append(hidden[j, period_pos[j].item(), :].numpy())
    finally:
        hook.remove()
    return np.stack(all_hidden, axis=0)


def get_correct_mask(
    model, tokenizer,
    prompts: list[str], labels: list[int],
    met_tok: int, lit_tok: int,
) -> np.ndarray:
    """Boolean mask: True where model's argmax(met_tok, lit_tok) matches the ground-truth label."""
    mask = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch  = prompts[i : i + BATCH_SIZE]
        b_lbls = labels[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(DEVICE)
        last_pos = inputs["attention_mask"].sum(dim=1) - 1
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits.cpu()
        for j, pos in enumerate(last_pos):
            lm   = logits[j, pos.item(), met_tok].item()
            ll   = logits[j, pos.item(), lit_tok].item()
            pred = 1 if lm > ll else 0
            mask.append(pred == b_lbls[j])
    return np.array(mask, dtype=bool)


#Activation patching IIA (layer sweep)
def patching_iia(
    model, tokenizer,
    prompts_met: list[str], prompts_lit: list[str],
    layer: int, met_tok: int, lit_tok: int,
) -> float:
    """Activation patching at period token (met→lit). Correct-only proxy for layer selection."""
    met_mask    = get_correct_mask(model, tokenizer, prompts_met, [1]*len(prompts_met), met_tok, lit_tok)
    lit_mask    = get_correct_mask(model, tokenizer, prompts_lit, [0]*len(prompts_lit), met_tok, lit_tok)
    prompts_met = [p for p, m in zip(prompts_met, met_mask) if m]
    prompts_lit = [p for p, m in zip(prompts_lit, lit_mask) if m]
    if not prompts_met or not prompts_lit:
        return 0.0
    H_lit   = extract_hidden(model, tokenizer, prompts_lit, layer)
    H_lit_t = torch.tensor(H_lit, dtype=torch.float32)
    n       = min(len(prompts_met), len(prompts_lit))
    prompts_met = prompts_met[:n]
    flipped = 0

    for i in range(0, n, BATCH_SIZE):
        batch_prompts = prompts_met[i : i + BATCH_SIZE]
        patch_vecs    = H_lit_t[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(DEVICE)
        last_pos   = inputs["attention_mask"].cpu().sum(dim=1) - 1
        period_pos = last_pos - PERIOD_OFFSET

        def make_hook(vecs, positions): #verride hidden activations
            def hook_fn(_module, _input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                for j, pos in enumerate(positions):
                    hidden[j, pos, :] = vecs[j].to(hidden.device, hidden.dtype) #patching
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            return hook_fn

        hook = model.model.layers[layer].register_forward_hook(
            make_hook(patch_vecs, period_pos.tolist())
        )
        try:
            with torch.no_grad():
                out = model(**inputs)
        finally:
            hook.remove()

        logits = out.logits.cpu()
        for j, pos in enumerate(last_pos):
            lm = logits[j, pos.item(), met_tok].item()
            ll = logits[j, pos.item(), lit_tok].item()
            if ll > lm:
                flipped += 1

    return flipped / n if n > 0 else 0.0


#Layer selection

#Helper layer selector
def kneedle_layer(layer_iia: dict) -> int:                                                                                                                            
    """                                                                                                                                                               
    Select the layer of maximum curvature in the IIA-vs-layer curve.                                                                                                  
    Normalises both axes to [0, 1] and finds argmax(y_n - x_n) — the point                                                                                            
    furthest above the diagonal connecting the first and last points.                                                                                                 
    Falls back to argmax IIA if the curve is flat.                                                                                                  
    """                                                                                                                                                               
    layers = sorted(layer_iia.keys())                                                                                                                                 
    iia    = np.array([layer_iia[l] for l in layers], dtype=float)                                                                                                    
                                                                                                                                                                        
    if iia.max() < 0.05:                                                                                                                                              
        print("  [Kneedle] Flat curve — falling back to max-IIA layer.")                                                                                              
        return layers[int(np.argmax(iia))]                                                                                                                            
                                                                                                                                                                         
    n = len(layers)                                                                                                                                                 
    x_n = np.arange(n, dtype=float) / max(n - 1, 1)                                                                                                                   
    y_r = iia.max() - iia.min()                                                                                                                                       
    y_n = (iia - iia.min()) / (y_r if y_r > 0 else 1.0)                                                                                                               
                                                                                                                                                                          
    knee_idx = int(np.argmax(y_n - x_n))                                                                                                                              
    print(f"  [Kneedle] Knee at layer {layers[knee_idx]}  "                                                                                                           
    f"IIA={iia[knee_idx]:.4f}  "                                                                                                                                
    f"(max={iia.max():.4f} at layer {layers[int(np.argmax(iia))]})")                                                                                            
    return layers[knee_idx]                                                                                                                                           

def select_layer(
    model, tokenizer, df: pd.DataFrame, met_tok: int, lit_tok: int,
) -> tuple[int, dict]:
    """Adjective-held-out activation patching IIA sweep, then kneedle on mean IIA curve."""
    adjectives = df["adj"].unique().tolist()
    rng = np.random.default_rng(42)
    layers = list(range(LAYER_START, min(LAYER_END + 1, len(model.model.layers))))
    layer_iias: dict[int, list[float]] = {l: [] for l in layers}

    for split_idx in range(N_HOLDOUT_SPLITS):
        held_out = rng.choice(adjectives, size=N_HOLDOUT_ADJS, replace=False).tolist()
        val_df   = df[df["adj"].isin(held_out)]

        met_val = val_df[val_df["label"] == 1]["prompt"].tolist()
        lit_val = val_df[val_df["label"] == 0]["prompt"].tolist()
        n_val   = min(len(met_val), len(lit_val))
        idx_m   = list(range(len(met_val))); rng.shuffle(idx_m)
        idx_l   = list(range(len(lit_val)));  rng.shuffle(idx_l)
        met_val = [met_val[i] for i in idx_m[:n_val]]
        lit_val = [lit_val[i] for i in idx_l[:n_val]]

        print(f"\nSplit {split_idx+1}/{N_HOLDOUT_SPLITS}  "
              f"held-out={held_out}  val_pairs={n_val}")

        for layer in layers:
            iia = patching_iia(model, tokenizer, met_val, lit_val, layer, met_tok, lit_tok)
            layer_iias[layer].append(iia)
            print(f"  L{layer:2d}: IIA={iia:.4f}", flush=True)

    sweep = {
        l: {"mean": float(np.mean(v)), "std": float(np.std(v)), "iias": v}
        for l, v in layer_iias.items()
    }
    mean_iia_per_layer = {l: sweep[l]["mean"] for l in sweep}
    best_layer = kneedle_layer(mean_iia_per_layer)
    print(f"\nBest layer: {best_layer}  "
          f"(IIA={sweep[best_layer]['mean']:.4f} ± {sweep[best_layer]['std']:.4f})")
    return best_layer, sweep


# DAS training 

def train_das(
    model, tokenizer,
    H_train: np.ndarray, labels_train: list[int], prompts_train: list[str],
    layer: int, k: int, met_tok: int, lit_tok: int,
    n_steps: int = None, seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Train DAS rotation. Injects at period token ('.' in test sentence)."""
    if n_steps is None:
        n_steps = N_STEPS
    fix_seeds(seed)
    d_model   = H_train.shape[1]
    rotation  = RotationMatrix(d_model, k).to(DEVICE)
    optimizer = Adam(rotation.parameters(), lr=1e-3)

    correct = get_correct_mask(model, tokenizer, prompts_train, labels_train, met_tok, lit_tok)
    c_idx   = np.where(correct)[0]
    H_train       = H_train[c_idx]
    labels_train  = [labels_train[i] for i in c_idx]
    prompts_train = [prompts_train[i] for i in c_idx]
    print(f"  correct-only filter: {len(c_idx)}/{len(correct)} train examples kept")

    rng = np.random.default_rng(seed)
    true_idx  = np.where(np.array(labels_train) == 1)[0]
    false_idx = np.where(np.array(labels_train) == 0)[0]
    rng.shuffle(true_idx); rng.shuffle(false_idx)
    n = min(len(true_idx), len(false_idx))
    if n == 0:
        raise ValueError("Need both metaphorical and literal examples.")

    H_met       = torch.tensor(H_train[true_idx[:n]],  dtype=torch.float32).to(DEVICE)
    H_lit       = torch.tensor(H_train[false_idx[:n]], dtype=torch.float32).to(DEVICE)
    prompts_met = [prompts_train[i] for i in true_idx[:n]]
    prompts_lit = [prompts_train[i] for i in false_idx[:n]]

    model.eval()
    for step in range(n_steps):
        idx = rng.integers(0, n, size=DAS_BATCH)

        # Alternate direction each step: met→lit on even, lit→met on odd
        if step % 2 == 0:
            h_base    = H_met[idx]
            h_src     = H_lit[idx]
            batch_prompts = [prompts_met[i] for i in idx]
            target_tok, other_tok = lit_tok, met_tok   # want output to flip to literal
        else:
            h_base    = H_lit[idx]
            h_src     = H_met[idx]
            batch_prompts = [prompts_lit[i] for i in idx]
            target_tok, other_tok = met_tok, lit_tok   # want output to flip to metaphor

        proj_base = rotation.project(h_base)
        proj_src  = rotation.project(h_src)
        patched   = h_base - proj_base + proj_src

        inputs   = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(DEVICE)
        last_pos   = inputs["attention_mask"].sum(dim=1) - 1
        period_pos = last_pos - PERIOD_OFFSET

        def make_hook(vecs, positions):
            def hook_fn(_module, _input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                for j, pos in enumerate(positions):
                    hidden[j, pos, :] = vecs[j].to(hidden.device, hidden.dtype)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            return hook_fn

        hook = model.model.layers[layer].register_forward_hook(
            make_hook(patched, period_pos.tolist())
        )
        try:
            out = model(**inputs)
        finally:
            hook.remove()

        logits     = out.logits
        loss_terms = []
        for j, pos in enumerate(last_pos):
            l_target = logits[j, pos.item(), target_tok]
            l_other  = logits[j, pos.item(), other_tok]
            loss_terms.append(l_other - l_target)   # minimise → target wins
        loss = torch.stack(loss_terms).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rotation.orthogonalise()

        if (step + 1) % 200 == 0:
            print(f"  step {step+1}/{n_steps}  loss={loss.item():.4f}", flush=True)

    R_np      = rotation.R.detach().cpu().numpy()
    train_iia = eval_das_iia(
        model, tokenizer, rotation, H_train, labels_train, prompts_train,
        layer, met_tok, lit_tok,
    )
    print(f"  Train IIA (k={k}, seed={seed}): {train_iia:.4f}")
    return R_np, train_iia


def eval_das_iia(
    model, tokenizer, rotation: RotationMatrix,
    H: np.ndarray, labels: list[int], prompts: list[str],
    layer: int, met_tok: int, lit_tok: int,
    return_per_sample: bool = False,
) -> float | tuple[float, list[int]]:
    """Bidirectional IIA for DAS rotation. Correct-only. Injects at period token.
    Averages met→lit and lit→met directions. Returns scalar.
    """
    correct_mask = get_correct_mask(model, tokenizer, prompts, labels, met_tok, lit_tok)
    labels_arr   = np.array(labels)
    rng          = np.random.default_rng(42)

    true_idx  = np.where((labels_arr == 1) & correct_mask)[0]
    false_idx = np.where((labels_arr == 0) & correct_mask)[0]
    rng.shuffle(true_idx); rng.shuffle(false_idx)
    n = min(len(true_idx), len(false_idx))

    if n == 0:
        return (0.0, []) if return_per_sample else 0.0

    true_idx  = true_idx[:n]
    false_idx = false_idx[:n]

    H_met       = torch.tensor(H[true_idx],  dtype=torch.float32).to(DEVICE)
    H_lit       = torch.tensor(H[false_idx], dtype=torch.float32).to(DEVICE)
    prompts_met = [prompts[i] for i in true_idx]
    prompts_lit = [prompts[i] for i in false_idx]

    rotation.eval()

    def _run_direction(h_base, h_src, batch_prompts, target_tok, other_tok):
        per: list[int] = []
        for i in range(0, n, DAS_BATCH):
            h_base_b = h_base[i : i + DAS_BATCH]
            h_src_b  = h_src[i : i + DAS_BATCH]
            batch_p  = batch_prompts[i : i + DAS_BATCH]
            with torch.no_grad():
                proj_base = rotation.project(h_base_b)
                proj_src  = rotation.project(h_src_b)
                patched   = h_base_b - proj_base + proj_src
            inputs   = tokenizer(
                batch_p, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(DEVICE)
            last_pos   = inputs["attention_mask"].sum(dim=1) - 1
            period_pos = last_pos - PERIOD_OFFSET

            def make_hook(vecs, positions):
                def hook_fn(_module, _input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    for j, pos in enumerate(positions):
                        hidden[j, pos, :] = vecs[j].to(hidden.device, hidden.dtype)
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                return hook_fn

            hook = model.model.layers[layer].register_forward_hook(
                make_hook(patched, period_pos.tolist())
            )
            try:
                with torch.no_grad():
                    out = model(**inputs)
            finally:
                hook.remove()
            logits = out.logits.cpu()
            for j, pos in enumerate(last_pos):
                lt = logits[j, pos.item(), target_tok].item()
                lo = logits[j, pos.item(), other_tok].item()
                per.append(1 if lt > lo else 0)
        return per

    # met→lit: patch lit subspace into met base, expect lit prediction
    per_ml = _run_direction(H_met, H_lit, prompts_met, lit_tok, met_tok)
    # lit→met: patch met subspace into lit base, expect met prediction
    per_lm = _run_direction(H_lit, H_met, prompts_lit, met_tok, lit_tok)

    per_sample = per_ml + per_lm
    iia = sum(per_sample) / (2 * n)
    if return_per_sample:
        return iia, per_sample
    return iia


# sweep over different values of the hyperparamter k in our DAS matrix.

def run_k_sweep(
    model, tokenizer,
    H_train: np.ndarray, labels_train: list[int], prompts_train: list[str],
    H_val:   np.ndarray, labels_val:   list[int], prompts_val:   list[str],
    layer: int, met_tok: int, lit_tok: int,
    label: str = "all",
) -> tuple[np.ndarray, int, dict]:
    d_model = H_train.shape[1]
    results: dict = {}

    print(f"\n=== DAS k sweep [{label}] at layer {layer} ===")
    all_seed_Rs: dict[int, list] = {}
    for k in K_VALUES:
        print(f"\n--- k={k} ---")
        seed_iias, seed_Rs = [], []
        for seed in SEEDS:
            R_np, _ = train_das(
                model, tokenizer, H_train, labels_train, prompts_train,
                layer, k, met_tok, lit_tok, seed=seed,
            )
            rotation = RotationMatrix(d_model, k)
            rotation.R = nn.Parameter(torch.tensor(R_np, dtype=torch.float32))
            rotation = rotation.to(DEVICE)
            val_iia, per_sample = eval_das_iia(
                model, tokenizer, rotation, H_val, labels_val, prompts_val,
                layer, met_tok, lit_tok, return_per_sample=True,
            )
            _, ci_lo_s, ci_hi_s = bootstrap_ci(per_sample)
            print(f"  seed={seed}  val_IIA={val_iia:.4f} "
                  f"[{ci_lo_s:.4f}, {ci_hi_s:.4f}]", flush=True)
            seed_iias.append(val_iia)
            seed_Rs.append((R_np, per_sample))

        pooled         = [x for _, s in seed_Rs for x in s]
        _, ci_lo, ci_hi = bootstrap_ci(pooled)
        std_iia        = float(np.std(seed_iias))
        results[k] = {
            "mean_iia":   round(float(np.mean(seed_iias)), 4),
            "std_iia":    round(std_iia, 4),
            "ci_lo":      round(ci_lo, 4),
            "ci_hi":      round(ci_hi, 4),
            "iias":       seed_iias,
            "per_sample": pooled,
        }
        all_seed_Rs[k] = seed_Rs
        print(f"  k={k}  IIA={results[k]['mean_iia']:.4f} ± {std_iia:.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]")

    best_k        = max(results, key=lambda k: results[k]["mean_iia"])
    best_seed_idx = int(np.argmax(results[best_k]["iias"]))
    best_R        = all_seed_Rs[best_k][best_seed_idx][0]
    print(f"\nBest k={best_k}  IIA={results[best_k]['mean_iia']:.4f}")
    return best_R, best_k, results


# Frobenius comparison

def projection_matrix(R: np.ndarray) -> np.ndarray:
    return R @ R.T


def frobenius_dist(P_a: np.ndarray, P_b: np.ndarray) -> float:
    return float(np.linalg.norm(P_a - P_b, "fro"))


def random_frobenius_baseline(d_model: int, k: int, n: int = 100) -> dict:
    rng = np.random.default_rng(0)
    dists = []
    for _ in range(n):
        A = np.linalg.qr(rng.standard_normal((d_model, d_model)))[0][:, :k]
        B = np.linalg.qr(rng.standard_normal((d_model, d_model)))[0][:, :k]
        dists.append(frobenius_dist(projection_matrix(A), projection_matrix(B)))
    mu = float(np.mean(dists))
    _, lo, hi = bootstrap_ci(dists)
    return {"mean": round(mu, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


# M3: Per-domain DAS + Frobenius + Transfer 

def run_m3(
    model, tokenizer,
    df: pd.DataFrame, H_all: np.ndarray,
    best_layer: int, best_k: int,
    best_R_all: np.ndarray,
    met_tok: int, lit_tok: int,
) -> dict:
    """Per-domain DAS, Frobenius pairwise distances, and transfer DAS (global R on each domain)."""
    domains = sorted(df["domain"].dropna().unique().tolist())
    d_model  = H_all.shape[1]
    domain_R: dict[str, np.ndarray] = {}
    domain_results:  dict[str, dict] = {}
    transfer_results: dict[str, dict] = {}

    rotation_all = RotationMatrix(d_model, best_k)
    rotation_all.R = nn.Parameter(torch.tensor(best_R_all, dtype=torch.float32))
    rotation_all = rotation_all.to(DEVICE)

    for domain in domains:
        print(f"\n--- Domain: {domain} ---")
        dom_mask    = (df["domain"] == domain).values
        dom_df      = df[df["domain"] == domain]
        dom_H       = H_all[dom_mask]
        dom_labels  = dom_df["label"].tolist()
        dom_prompts = dom_df["prompt"].tolist()

        # Adjective-held-out split: hold out 1 adjective
        n_dom        = len(dom_labels)
        dom_adjs     = sorted(dom_df["adj"].unique().tolist())
        rng_dom      = np.random.default_rng(42)
        held_out_adj = rng_dom.choice(dom_adjs)
        tr_mask      = dom_df["adj"].values != held_out_adj
        va_mask      = dom_df["adj"].values == held_out_adj
        tr_idx       = np.where(tr_mask)[0]
        va_idx       = np.where(va_mask)[0]
        print(f"  held-out adj: '{held_out_adj}'  "
              f"train={len(tr_idx)}  val={len(va_idx)}")

        H_d_tr   = dom_H[tr_idx]
        lbl_d_tr = [dom_labels[i] for i in tr_idx]
        pr_d_tr  = [dom_prompts[i] for i in tr_idx]
        H_d_va   = dom_H[va_idx]
        lbl_d_va = [dom_labels[i] for i in va_idx]
        pr_d_va  = [dom_prompts[i] for i in va_idx]

        # Per-domain DAS (train own R)
        seed_iias, seed_Rs = [], []
        for seed in SEEDS:
            R_np, _ = train_das(
                model, tokenizer, H_d_tr, lbl_d_tr, pr_d_tr,
                best_layer, best_k, met_tok, lit_tok, seed=seed,
            )
            rot = RotationMatrix(d_model, best_k)
            rot.R = nn.Parameter(torch.tensor(R_np, dtype=torch.float32))
            rot = rot.to(DEVICE)
            val_iia, per_sample = eval_das_iia(
                model, tokenizer, rot, H_d_va, lbl_d_va, pr_d_va,
                best_layer, met_tok, lit_tok, return_per_sample=True,
            )
            seed_iias.append(val_iia)
            seed_Rs.append((R_np, per_sample))
            print(f"  seed={seed}  val_IIA={val_iia:.4f}", flush=True)

        pooled         = [x for _, s in seed_Rs for x in s]
        _, ci_lo, ci_hi = bootstrap_ci(pooled)
        std_iia        = float(np.std(seed_iias))
        domain_results[domain] = {
            "mean_iia":   round(float(np.mean(seed_iias)), 4),
            "std_iia":    round(std_iia, 4),
            "ci_lo":      round(ci_lo, 4),
            "ci_hi":      round(ci_hi, 4),
            "iias":       seed_iias,
            "per_sample": pooled,
            "n":          n_dom,
        }
        print(f"  {domain}  per-domain IIA={domain_results[domain]['mean_iia']:.4f} ± {std_iia:.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]")

        best_seed_idx = int(np.argmax(seed_iias))
        domain_R[domain] = seed_Rs[best_seed_idx][0]
        np.save(
            f"results/R_metaphor_final_{domain}_k{best_k}_{MODEL_TAG}.npy",
            domain_R[domain],
        )

        # Transfer DAS: evaluate global R on this domain's val data
        t_iia, t_per_sample = eval_das_iia(
            model, tokenizer, rotation_all, H_d_va, lbl_d_va, pr_d_va,
            best_layer, met_tok, lit_tok, return_per_sample=True,
        )
        _, t_lo, t_hi = bootstrap_ci(t_per_sample)
        transfer_results[domain] = {
            "mean_iia":   round(t_iia, 4),
            "ci_lo":      round(t_lo, 4),
            "ci_hi":      round(t_hi, 4),
            "per_sample": t_per_sample,
        }
        print(f"  {domain}  transfer IIA (global R)={t_iia:.4f} [{t_lo:.4f}, {t_hi:.4f}]")

    # Frobenius pairwise distances between domain projection matrices
    print("\n=== Frobenius pairwise distances ===")
    frob_matrix: dict[str, dict[str, float]] = {}
    all_dists = []
    for i, d_a in enumerate(domains):
        frob_matrix[d_a] = {}
        for j, d_b in enumerate(domains):
            if d_a == d_b:
                frob_matrix[d_a][d_b] = 0.0
                continue
            P_a  = projection_matrix(domain_R[d_a])
            P_b  = projection_matrix(domain_R[d_b])
            dist = frobenius_dist(P_a, P_b)
            frob_matrix[d_a][d_b] = round(dist, 4)
            if i < j:
                all_dists.append(dist)
                print(f"  {d_a} vs {d_b}: {dist:.4f}")

    frob_mean = float(np.mean(all_dists))
    _, frob_lo, frob_hi = bootstrap_ci(all_dists)
    random_baseline = random_frobenius_baseline(d_model, best_k)
    print(f"\nMean pairwise Frobenius: {frob_mean:.4f} [{frob_lo:.4f}, {frob_hi:.4f}]")
    print(f"Random baseline:         {random_baseline['mean']:.4f} "
          f"[{random_baseline['ci_lo']:.4f}, {random_baseline['ci_hi']:.4f}]")

    return {
        "per_domain_das": domain_results,
        "transfer_das":   transfer_results,
        "frobenius": {
            "pairwise_matrix": frob_matrix,
            "mean_all_pairs":  round(frob_mean, 4),
            "ci_lo":           round(frob_lo,   4),
            "ci_hi":           round(frob_hi,   4),
            "random_baseline": random_baseline,
        },
    }


# main

def main():
    fix_seeds(42)
    os.makedirs("results", exist_ok=True)

    model, tokenizer = load_model(MODEL_NAME)
    met_tok, lit_tok = get_met_lit_toks(tokenizer)
    print(f"\nmet_tok: '{tokenizer.decode([met_tok])}' ({met_tok})")
    print(f"lit_tok: '{tokenizer.decode([lit_tok])}' ({lit_tok})")

    df = load_data()
    prompts = df["prompt"].tolist()

    if M2_RESULTS:
        # M3-only mode: load existing M2 results, skip M1+M2
        print(f"\nM3-only mode — loading M2 results from {M2_RESULTS}")
        with open(M2_RESULTS) as f:
            m2 = json.load(f)
        best_layer  = m2["best_layer"]
        best_k_all  = m2["best_k"]
        k_results_all = {int(k): v for k, v in m2["full_das"].items()}
        layer_sweep = {int(k): v for k, v in m2.get("layer_sweep", {}).items()}
        print(f"  best_layer={best_layer}  best_k={best_k_all}")

        R_path = f"results/R_metaphor_final_k{best_k_all}_{MODEL_TAG}.npy"
        if not os.path.exists(R_path):
            raise FileNotFoundError(
                f"Global R matrix not found: {R_path}\n"
                "Copy it from Snellius before running M3-only mode."
            )
        best_R_all = np.load(R_path)
        print(f"  Loaded R_all from {R_path}")

        print(f"\nExtracting hidden states at layer {best_layer} ...")
        H_all = extract_hidden(model, tokenizer, prompts, best_layer)

    else:
        # Full pipeline: M1 → M2 → M3 
        print("\n" + "="*60)
        print("M1: Layer selection (adjective-held-out activation patching IIA — kneedle)")
        print("="*60)
        if FORCE_LAYER >= 0:
            best_layer  = FORCE_LAYER
            layer_sweep = {}
            print(f"\nFORCE_LAYER={FORCE_LAYER} — skipping layer sweep.")
        else:
            best_layer, layer_sweep = select_layer(model, tokenizer, df, met_tok, lit_tok)

        print(f"\nExtracting hidden states at layer {best_layer} (period token) ...")
        H_all  = extract_hidden(model, tokenizer, prompts, best_layer)
        labels = df["label"].tolist()

        # M3 held-out adjectives (one per domain, seed=42) serve as the unified val set:
        # global M2 val IIA  = pooled across all domains (same split)
        # M3 transfer IIA    = per-domain slice of the same held-out set
        m3_held_out: set[str] = set()
        for _domain in sorted(df["domain"].dropna().unique()):
            _dom_adjs = sorted(df[df["domain"] == _domain]["adj"].unique().tolist())
            _rng_m3   = np.random.default_rng(42)
            m3_held_out.add(str(_rng_m3.choice(_dom_adjs)))

        train_mask = ~df["adj"].isin(m3_held_out)
        val_mask   =  df["adj"].isin(m3_held_out)

        H_train       = H_all[train_mask.values]
        labels_train  = df[train_mask]["label"].tolist()
        prompts_train = df[train_mask]["prompt"].tolist()
        H_val         = H_all[val_mask.values]
        labels_val    = df[val_mask]["label"].tolist()
        prompts_val   = df[val_mask]["prompt"].tolist()

        print(f"Train: {len(labels_train)} pairs  Val: {len(labels_val)} pairs  "
              f"(held-out adjectives: {sorted(m3_held_out)})")

        print("\n" + "="*60)
        print("M2: DAS on all pairs (k sweep, 3 seeds each)")
        print("="*60)
        best_R_all, best_k_all, k_results_all = run_k_sweep(
            model, tokenizer,
            H_train, labels_train, prompts_train,
            H_val,   labels_val,   prompts_val,
            best_layer, met_tok, lit_tok,
            label="all_pairs",
        )
        np.save(f"results/R_metaphor_final_k{best_k_all}_{MODEL_TAG}.npy", best_R_all)
        print(f"Saved R_all → results/R_metaphor_final_k{best_k_all}_{MODEL_TAG}.npy")

    # M3: Per-domain DAS + Frobenius + Transfer
    print("\n" + "="*60)
    print("M3: Per-domain DAS + Frobenius + Transfer DAS")
    print("="*60)
    m3_out = run_m3(
        model, tokenizer, df, H_all,
        best_layer, best_k_all, best_R_all,
        met_tok, lit_tok,
    )

    # Save 
    out = {
        "model":          MODEL_NAME,
        "n_shots":        N_SHOTS,
        "best_layer":     best_layer,
        "best_k":         best_k_all,
        "layer_sweep":    {str(l): v for l, v in layer_sweep.items()},
        "full_das":       {str(k): v for k, v in k_results_all.items()},
        **m3_out,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
