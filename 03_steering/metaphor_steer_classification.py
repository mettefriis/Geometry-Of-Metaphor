"""
Steer wrong predictions to the correct label with mass-mean and with the global DAS
subspace, at the "." token of the DAS layer. Class means come from correctly
predicted examples. 100 wrongly predicted examples (50 per class) are steered.
A flip = the steered prediction equals the true label.

Needs the DAS outputs (best layer, R matrix) from 01_domain_general_das.
See jobs/run_steer_class_final_<model>.job for the exact settings.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


#helpers

def fix_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


PERIOD_OFFSET = 4
BATCH_SIZE_EXTRACT = int(os.environ.get("BATCH_SIZE", "16"))

SENTENCE_FRAME = "The {noun} was {adj}."

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

_N_SHOTS   = int(os.environ.get("N_SHOTS", "5"))
_DATA_PATH = os.environ.get(
    "METAPHOR_DATA",
    "data/gut-2016-new.csv",
)


def _make_prompt(adj: str, noun: str, few_shot_prefix: str = "") -> str:
    test_sent = SENTENCE_FRAME.format(noun=noun, adj=adj)
    return f"{few_shot_prefix}{test_sent} This sentence is:"


def _build_few_shot_prefix(df: pd.DataFrame, exclude_adj: str, n_shots: int,
                            rng: random.Random) -> str:
    pool      = df[df["adj"] != exclude_adj]
    met_rows  = pool[pool["label"] == 1].to_dict("records")
    lit_rows  = pool[pool["label"] == 0].to_dict("records")
    rng.shuffle(met_rows); rng.shuffle(lit_rows)
    n_met     = n_shots // 2; n_lit = n_shots - n_met
    examples  = met_rows[:n_met] + lit_rows[:n_lit]
    rng.shuffle(examples)
    lines = []
    for ex in examples:
        sent  = SENTENCE_FRAME.format(noun=ex["noun"], adj=ex["adj"])
        label = "metaphor" if ex["label"] == 1 else "literal"
        lines.append(f"{sent} This sentence is: {label}.")
    return "\n".join(lines) + "\n"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH)
    df["adj"]    = df["expression"].str.strip()
    df["noun"]   = df["tenor"].str.strip()
    df["label"]  = (df["label"].str.strip() == "m").astype(int)
    df["domain"] = df["adj"].map(CMT_DOMAIN)
    rng = random.Random(42)
    prefix_cache = {adj: _build_few_shot_prefix(df, adj, _N_SHOTS, rng)
                    for adj in df["adj"].unique()}
    df["prompt"] = [
        _make_prompt(r.adj, r.noun, prefix_cache[r.adj])
        for r in df.itertuples()
    ]
    print(f"Loaded {len(df)} pairs  "
          f"({df['label'].sum()} metaphorical, {(df['label']==0).sum()} literal)")
    return df


def extract_hidden(model, tokenizer, prompts: list[str], layer: int) -> np.ndarray:
    """Extract residual stream at period token at given layer."""
    all_hidden = []
    captured   = {}

    def hook_fn(_module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().cpu().float()

    _device = next(model.parameters()).device
    hook = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        for i in range(0, len(prompts), BATCH_SIZE_EXTRACT):
            batch  = prompts[i : i + BATCH_SIZE_EXTRACT]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(_device)
            with torch.no_grad():
                model(**inputs)
            hidden     = captured["h"]
            period_pos = inputs["attention_mask"].cpu().sum(dim=1) - 1 - PERIOD_OFFSET
            for j in range(len(batch)):
                all_hidden.append(hidden[j, period_pos[j].item(), :].numpy())
    finally:
        hook.remove()
    return np.stack(all_hidden, axis=0)


# ---------------------------------------------------------------------------

# Config 

MODEL_NAME       = os.environ.get("DAS_MODEL",        "meta-llama/Llama-3.1-8B-Instruct")
R_GLOBAL_PATH    = os.environ.get("R_GLOBAL_PATH",    "")
BEST_LAYER       = int(os.environ.get("BEST_LAYER",    "16"))
ALPHAS           = [float(a) for a in os.environ.get("ALPHAS", "1").split(",")]
SEED             = int(os.environ.get("SEED",          "42"))
N_STEER          = int(os.environ.get("N_STEER",       "100"))
MODEL_TAG        = MODEL_NAME.replace("/", "_").replace("-", "_").lower()
RESULTS_FILE     = os.environ.get(
    "RESULTS_FILE",
    f"results/results_metaphor_steer_class_final_{MODEL_TAG}.json",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Label indices: 0=literal, 1=metaphor
LABEL_STR = {0: "literal", 1: "metaphor"}


# Class means

def compute_class_means(model, tokenizer, prompts_met: list, prompts_lit: list, layer: int):
    """Class means from correctly predicted samples only."""
    H_met  = extract_hidden(model, tokenizer, prompts_met, layer)
    H_lit  = extract_hidden(model, tokenizer, prompts_lit, layer)
    mu_met = torch.tensor(H_met.mean(axis=0), dtype=torch.float32).to(DEVICE)
    mu_lit = torch.tensor(H_lit.mean(axis=0), dtype=torch.float32).to(DEVICE)
    print(f"  μ_met from {len(H_met)} correct metaphor examples")
    print(f"  μ_lit from {len(H_lit)} correct literal examples")
    return mu_met, mu_lit


# R matrix loading 

def load_projection(path: str) -> torch.Tensor:
    R_np = np.load(path)
    P_np = R_np @ R_np.T
    print(f"  R shape: {R_np.shape}")
    return torch.tensor(P_np, dtype=torch.float32).to(DEVICE)


# Single-forward-pass prediction 

def get_label_token_ids(tokenizer) -> dict[int, int]:
    lit_tok = tokenizer(" literal",  add_special_tokens=False).input_ids[0]
    met_tok = tokenizer(" metaphor", add_special_tokens=False).input_ids[0]
    return {0: lit_tok, 1: met_tok}


def predict(model, tokenizer, prompt: str, label_ids: dict[int, int],
            patch_fn=None) -> int:
    """Single forward pass. Returns predicted label (0 or 1).
    If patch_fn is None, no hook is applied (baseline).
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024,
    ).to(DEVICE)

    handle = None
    if patch_fn is not None:
        def hook(_module, _input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            period_idx = hidden.shape[1] - 1 - PERIOD_OFFSET
            hidden[:, period_idx, :] = patch_fn(hidden[:, period_idx, :].squeeze(0)).unsqueeze(0) #patch_fn is a variable holding the oputput of make_mass_mean_fn or make_das_fn
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        handle = model.model.layers[BEST_LAYER].register_forward_hook(hook)

    try:
        with torch.no_grad():
            out = model(**inputs)
    finally:
        if handle is not None:
            handle.remove()

    logits = out.logits[0, -1, :] # tensor of vocab size
    score_lit = logits[label_ids[0]].item()
    score_met = logits[label_ids[1]].item()
    return 0 if score_lit > score_met else 1


# Steer functions 

def make_mass_mean_fn(steer_vec: torch.Tensor, alpha: float):
    shift = (alpha * steer_vec)
    def fn(h): return h + shift.to(h.device, h.dtype)
    return fn #returns the function computing the steered vector


def make_das_fn(mu_target: torch.Tensor, P: torch.Tensor, alpha: float):
    def fn(h):
        dev, dt = h.device, h.dtype
        return h - alpha * (P.to(dev, dt) @ (h - mu_target.to(dev, dt)))
    return fn #returns the function computing the steered vector


# Main
def main():
    fix_seeds(SEED)
    os.makedirs("results", exist_ok=True)

    assert R_GLOBAL_PATH, "Set R_GLOBAL_PATH" #guard - crash if empty

    model, tokenizer = load_model(MODEL_NAME)
    label_ids = get_label_token_ids(tokenizer)
    print(f"  Label token ids: literal={label_ids[0]}, metaphor={label_ids[1]}")

    df = load_data()

    print(f"\nLoading global R: {R_GLOBAL_PATH}")
    P_global = load_projection(R_GLOBAL_PATH)

    print(f"\nModel: {MODEL_NAME}  layer={BEST_LAYER}  alphas={ALPHAS}\n")

    # Run baseline on all samples; split into correct (for class means) and wrong (for steering)
    print("Running baseline on all samples ...")
    correct_prompts_met, correct_prompts_lit = [], []
    wrong_met, wrong_lit = [], []
    for i, row in df.iterrows():
        adj, noun, true_label, prompt = row["adj"], row["noun"], row["label"], row["prompt"]
        base_pred = predict(model, tokenizer, prompt, label_ids, patch_fn=None)
        print(f"  [{i+1}/{len(df)}] {adj}+{noun}  true={LABEL_STR[true_label]}  "
              f"pred={LABEL_STR[base_pred]}  {'✗' if base_pred != true_label else '✓'}", flush=True)
        if base_pred == true_label:
            (correct_prompts_met if true_label == 1 else correct_prompts_lit).append(prompt)
        else:
            entry = {"adj": adj, "noun": noun, "label": true_label,
                     "domain": row.get("domain", "UNKNOWN"),
                     "prompt": prompt, "base_pred": base_pred}
            (wrong_met if true_label == 1 else wrong_lit).append(entry)

    print(f"\nCorrect: {len(correct_prompts_met)} metaphor, {len(correct_prompts_lit)} literal")
    print(f"Wrong:   {len(wrong_met)} metaphor, {len(wrong_lit)} literal "
          f"(total {len(wrong_met)+len(wrong_lit)}/{len(df)})")

    # Class means from correctly predicted samples only
    print(f"\nComputing class means at layer {BEST_LAYER} ...")
    mu_met, mu_lit = compute_class_means(
        model, tokenizer, correct_prompts_met, correct_prompts_lit, BEST_LAYER
    )

    # Balanced sample of N_STEER wrong examples
    rng = random.Random(SEED)
    rng.shuffle(wrong_met); rng.shuffle(wrong_lit)
    n_each = N_STEER // 2
    samples_with_base = wrong_met[:n_each] + wrong_lit[:N_STEER - n_each]
    rng.shuffle(samples_with_base)
    print(f"Sampled {len(samples_with_base)} "
          f"({min(n_each, len(wrong_met))} met-wrong, "
          f"{min(N_STEER - n_each, len(wrong_lit))} lit-wrong)\n")

    # Sweep alphas
    by_alpha: dict[str, list] = {}
    for alpha in ALPHAS:
        print(f"\n{'='*60}\nAlpha = {alpha}\n{'='*60}")
        alpha_outputs = []
        mm_flips, das_flips = 0, 0

        for s in samples_with_base:
            adj        = s["adj"]
            noun       = s["noun"]
            true_label = s["label"]
            prompt     = s["prompt"]

            # Steer toward correct (true) label
            mu_target  = mu_met if true_label == 1 else mu_lit
            mu_wrong   = mu_lit if true_label == 1 else mu_met
            steer_vec  = mu_target - mu_wrong

            mm_pred  = predict(model, tokenizer, prompt, label_ids,
                               make_mass_mean_fn(steer_vec, alpha))
            das_pred = predict(model, tokenizer, prompt, label_ids,
                               make_das_fn(mu_target, P_global, alpha))

            mm_flip  = int(mm_pred  == true_label)
            das_flip = int(das_pred == true_label)
            mm_flips  += mm_flip
            das_flips += das_flip

            print(f"  {adj}+{noun}  true={LABEL_STR[true_label]}  "
                  f"base={LABEL_STR[s['base_pred']]}  "
                  f"mm={LABEL_STR[mm_pred]}({'✓' if mm_flip else '✗'})  "
                  f"das={LABEL_STR[das_pred]}({'✓' if das_flip else '✗'})", flush=True)

            alpha_outputs.append({
                "adj":        adj,
                "noun":       noun,
                "domain":     s.get("domain", "UNKNOWN"),
                "true_label": LABEL_STR[true_label],
                "base_pred":  LABEL_STR[s["base_pred"]],
                "mm_pred":    LABEL_STR[mm_pred],
                "das_pred":   LABEL_STR[das_pred],
                "mm_flip":    mm_flip,
                "das_flip":   das_flip,
            })

        n = len(alpha_outputs)
        print(f"\n  alpha={alpha}  MM flip: {mm_flips}/{n} = {mm_flips/n:.3f}  "
              f"DAS flip: {das_flips}/{n} = {das_flips/n:.3f}")
        by_alpha[str(alpha)] = alpha_outputs

    result = {
        "model":       MODEL_NAME,
        "layer":       BEST_LAYER,
        "alphas":      ALPHAS,
        "n_total":          len(df),
        "n_correct_met":    len(correct_prompts_met),
        "n_correct_lit":    len(correct_prompts_lit),
        "n_wrong_met":      len(wrong_met),
        "n_wrong_lit":      len(wrong_lit),
        "n_samples":   len(samples_with_base),
        "by_alpha":    by_alpha,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {RESULTS_FILE}")


if __name__ == "__main__":
    main()
