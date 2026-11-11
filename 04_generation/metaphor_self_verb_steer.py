"""
Steer free generation toward metaphorical or literal meaning. The model is asked
for the meaning of the adjective ("The {noun} was {adj}. The meaning of '{adj}' is:")
and generates with no steering, with mass-mean steering, and with DAS steering at
the DAS layer: at the "." in the prompt for the first generated token and then at every generated token following the design from Li et al. 2023. 
Run for 100
literal sentences steered to metaphorical and 100 metaphorical steered to literal, at alpha = 1, 2, 5, 10. 
The outputs are scored by 05_judge/metaphor_llm_judge_local.py.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import torch

# helpers

def fix_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
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


PERIOD_OFFSET      = 4
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


def _build_few_shot_prefix(df, exclude_adj: str, n_shots: int,
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


def load_data() -> "pd.DataFrame":
    import pandas as pd
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


def extract_hidden(model, tokenizer, prompts: list, layer: int) -> np.ndarray:
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


#configuations

MODEL_NAME    = os.environ.get("DAS_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
R_MATRIX_PATH = os.environ.get("R_MATRIX_PATH", "")
BEST_LAYER    = int(os.environ.get("BEST_LAYER",    "16"))
N_SHOTS       = int(os.environ.get("N_SHOTS",       "5"))
ALPHAS        = [float(a) for a in os.environ.get("ALPHAS", "1").split(",")]
STEER_DIRS    = os.environ.get("STEER_DIRS", "met,lit").split(",")
SEED          = int(os.environ.get("SEED",           "42"))
N_GEN_EXAMPLES = int(os.environ.get("N_GEN_EXAMPLES", "10"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "60"))
MODEL_TAG     = MODEL_NAME.replace("/", "_").replace("-", "_").lower()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCE_FRAME = "The {noun} was {adj}."

# Fixed 5-shot meaning-format prefix (3 metaphorical, 2 literal examples)
MEANING_PREFIX = (
    "The welcome was warm. The meaning of 'warm' is: affectionate and emotionally inviting.\n"
    "The lamp was bright. The meaning of 'bright' is: emitting a strong, clear light.\n"
    "The novel was deep. The meaning of 'deep' is: intellectually complex and rich in meaning.\n"
    "The stone was smooth. The meaning of 'smooth' is: having an even surface with no roughness.\n"
    "The defeat was bitter. The meaning of 'bitter' is: painful, hard to accept, and leaving resentment.\n"
)


# helpers
def make_gen_prompt(adj: str, noun: str) -> str:
    test_sent = SENTENCE_FRAME.format(noun=noun, adj=adj)
    return f"{MEANING_PREFIX}{test_sent} The meaning of '{adj}' is:"


# Class-mean extraction 

def compute_class_means(
    model, tokenizer, df, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute μ_met and μ_lit at the period token position.
    Matches the position the R matrix was trained on."""
    prompts = df["prompt"].tolist()
    labels  = df["label"].tolist()
    H     = extract_hidden(model, tokenizer, prompts, layer)   # extracts at period_pos
    H_met = H[[i for i, l in enumerate(labels) if l == 1]]
    H_lit = H[[i for i, l in enumerate(labels) if l == 0]]
    mu_met = torch.tensor(H_met.mean(axis=0), dtype=torch.float32).to(DEVICE)
    mu_lit = torch.tensor(H_lit.mean(axis=0), dtype=torch.float32).to(DEVICE)
    print(f"  μ_met from {len(H_met)} examples, μ_lit from {len(H_lit)}")
    return mu_met, mu_lit


# Persistent-hook generation 

def _find_last_period_pos(input_ids: torch.Tensor, tokenizer) -> int:
    """Return the index of the last '.' token in input_ids (fallback: last token)."""
    period_ids = set(tokenizer.encode(".", add_special_tokens=False))
    positions = [i for i, t in enumerate(input_ids.tolist()) if t in period_ids]
    return positions[-1] if positions else input_ids.shape[0] - 1


def generate_with_hook(
    model, tokenizer, prompt: str, layer: int, patch_fn, max_new_tokens: int
) -> str:
    """
    Generates text with patch_fn applied at `layer` on every forward step.
    Prefill: patches at the last '.' in the prompt (same position as R training).
    Decode: patches at idx=0.
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024,
    ).to(DEVICE)

    prefill_period_idx = _find_last_period_pos(inputs["input_ids"][0], tokenizer)

    def hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        if hidden.shape[1] == 1:
            # decode step: KV cache active, only the new token is present
            idx = 0
        else:
            # prefill: patch at last '.' (matches period-token R training position)
            idx = prefill_period_idx
        hidden[:, idx, :] = patch_fn(hidden[:, idx, :].squeeze(0)).unsqueeze(0)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()

    new_ids  = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024,
    ).to(DEVICE)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# Patch functions

def make_mass_mean_fn(steer_vec: torch.Tensor, alpha: float):
    """Add α * (μ_target - μ_source) to h."""
    shift = alpha * steer_vec
    def patch_fn(h: torch.Tensor) -> torch.Tensor:
        return h + shift.to(h.device, h.dtype)
    return patch_fn


def make_das_fn(mu_target: torch.Tensor, P_t: torch.Tensor, alpha: float):
    """Project h onto subspace component of mu_target: h → h - α*P@(h - μ_target)."""
    def patch_fn(h: torch.Tensor) -> torch.Tensor:
        dev, dt = h.device, h.dtype
        delta = P_t.to(dev, dt) @ (h - mu_target.to(dev, dt))
        return h - alpha * delta
    return patch_fn


#Main

def main():
    fix_seeds(SEED)
    os.makedirs("results", exist_ok=True)

    assert R_MATRIX_PATH, "R_MATRIX_PATH must be set (path to .npy R matrix)"
    for sd in STEER_DIRS:
        assert sd in ("met", "lit"), f"STEER_DIRS must be 'met' and/or 'lit', got '{sd}'"

    print(f"Model:      {MODEL_NAME}")
    print(f"Layer:      {BEST_LAYER}")
    print(f"R matrix:   {R_MATRIX_PATH}")
    print(f"Alphas:     {ALPHAS}")
    print(f"Steer dirs: {STEER_DIRS}")
    print(f"Examples:   {N_GEN_EXAMPLES}")
    print(f"Max tokens: {MAX_NEW_TOKENS}\n")

    model, tokenizer = load_model(MODEL_NAME)

    # Load R matrix (once for all runs)
    R_np = np.load(R_MATRIX_PATH)
    P_np = R_np @ R_np.T
    P_t  = torch.tensor(P_np, dtype=torch.float32).to(DEVICE)
    print(f"R shape: {R_np.shape}  (k={R_np.shape[1]})\n")

    # Load data and build prompts (once)
    df = load_data()

    # Compute class means at the target layer (once)
    print(f"Computing class means at layer {BEST_LAYER} ...")
    mu_met, mu_lit = compute_class_means(model, tokenizer, df, BEST_LAYER)
    print()

    # Loop over all (steer_dir, alpha) combinations
    for steer_dir in STEER_DIRS:
        if steer_dir == "met":
            steer_vec  = mu_met - mu_lit
            mu_target  = mu_met
            src_label  = 0
            label_name = ("literal", "metaphorical")
        else:
            steer_vec  = mu_lit - mu_met
            mu_target  = mu_lit
            src_label  = 1
            label_name = ("metaphorical", "literal")

        # Sample test examples once per direction (same seed → reproducible)
        rng = random.Random(SEED)
        pool = df[df["label"] == src_label].to_dict("records")
        rng.shuffle(pool)
        test_examples = pool[:N_GEN_EXAMPLES]
        print(f"[dir={steer_dir}] {len(test_examples)} {label_name[0]} examples → {label_name[1]}")

        for alpha in ALPHAS:
            alpha_int = int(alpha) if alpha == int(alpha) else alpha
            results_file = (
                f"results/results_metaphor_generation_final_"
                f"{steer_dir}_a{alpha_int}_{MODEL_TAG}.json"
            )

            print(f"\n{'='*70}")
            print(f"  steer_dir={steer_dir}  alpha={alpha}  → {results_file}")
            print(f"{'='*70}")

            mass_mean_fn = make_mass_mean_fn(steer_vec, alpha)
            das_fn       = make_das_fn(mu_target, P_t, alpha)

            all_outputs = []

            for i, ex in enumerate(test_examples):
                adj, noun = ex["adj"], ex["noun"]
                prompt    = make_gen_prompt(adj, noun)

                print(f"\n[{i+1}/{len(test_examples)}] '{adj}' + '{noun}'  "
                      f"(true label: {label_name[0]}, domain: {ex.get('domain', '?')})")

                base_out = generate_baseline(model, tokenizer, prompt, MAX_NEW_TOKENS)
                print(f"  BASELINE  : {base_out}")

                mm_out = generate_with_hook(
                    model, tokenizer, prompt, BEST_LAYER, mass_mean_fn, MAX_NEW_TOKENS
                )
                print(f"  MASS_MEAN : {mm_out}")

                das_out = generate_with_hook(
                    model, tokenizer, prompt, BEST_LAYER, das_fn, MAX_NEW_TOKENS
                )
                print(f"  DAS       : {das_out}")

                all_outputs.append({
                    "adj":        adj,
                    "noun":       noun,
                    "domain":     ex.get("domain"),
                    "true_label": label_name[0],
                    "steer_dir":  label_name[1],
                    "baseline":   base_out,
                    "mass_mean":  mm_out,
                    "das":        das_out,
                })

            output = {
                "model":      MODEL_NAME,
                "layer":      BEST_LAYER,
                "alpha":      alpha,
                "steer_dir":  steer_dir,
                "r_matrix":   R_MATRIX_PATH,
                "n_examples": len(all_outputs),
                "examples":   all_outputs,
            }
            with open(results_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved → {results_file}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
