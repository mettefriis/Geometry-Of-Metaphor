"""
Score the outputs of metaphor_self_verb_steer.py with an LLM judge (Qwen2.5-72B-Instruct).
The judge labels the first sentence of each generated meaning as literal or
metaphorical. A first sentence that is empty or repeats a word 3+ times is marked
degenerate. A flip = the judge label equals the steering target.

Per file (model x direction x alpha) it reports the raw flip rates, and filtered flip
rates: only examples whose unsteered output the judge labels correctly, excluding the
5 adjectives in the few-shot prefix. Degenerate outputs counted as no flip.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config 

JUDGE_MODEL      = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
RESULTS_GLOB     = os.environ.get("RESULTS_GLOB",
                       "results/results_metaphor_generation_final_*.json")
RESULTS_OUT_DIR  = os.environ.get("RESULTS_OUT_DIR", "results/judge")
REPEAT_THRESHOLD = int(os.environ.get("REPEAT_THRESHOLD", "3"))
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

# Adjectives that appear in the fixed few-shot prefix — excluded from filtered metrics
FEWSHOT_ADJS = {"warm", "bright", "deep", "smooth", "bitter"}

SYSTEM_PROMPT = (
    "You are classifying whether a meaning description conveys a literal or "
    "metaphorical sense of an adjective.\n\n"
    "A LITERAL meaning describes a physical, concrete, or sensory property "
    "(e.g. temperature, texture, weight, brightness).\n"
    "A METAPHORICAL meaning describes an abstract, figurative, or emotional "
    "property (e.g. mood, personality, intellectual quality).\n\n"
    "Reply with exactly one word: either \"literal\" or \"metaphorical\". "
    "No explanation. Do not repeat the sentence."
)

USER_TEMPLATE = (
    'Sentence: "{sentence}"\n'
    "Adjective: '{adj}'\n"
    'Meaning description: "{completion}"\n\n'
    "Is this meaning literal or metaphorical?"
)


# Text utilities

def extract_first_sentence(text: str) -> str:
    """Return only the first sentence of text."""
    text = text.strip()
    match = re.search(r'(?<=[.!?])\s', text)
    if match:
        return text[:match.start()].strip()
    # Fallback: first newline
    first_line = text.split("\n")[0].strip()
    return first_line if first_line else text


def is_degenerate(text: str) -> bool:
    """True if text is empty or any word repeats >= REPEAT_THRESHOLD times."""
    if not text or not text.strip():
        return True
    words = text.lower().split()
    if not words:
        return True
    most_common_count = Counter(words).most_common(1)[0][1]
    return most_common_count >= REPEAT_THRESHOLD


# Judge load
def load_judge(model_name: str):
    print(f"Loading judge model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Judge loaded.\n")
    return model, tokenizer


def judge_completion(
    model, tokenizer, sentence: str, adj: str, completion: str
) -> str:
    """
    Returns 'literal', 'metaphorical', or 'degenerate'.
    Only the first sentence of completion is judged.
    """
    first = extract_first_sentence(completion)
    if is_degenerate(first):
        return "degenerate"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(
            sentence=sentence, adj=adj, completion=first
        )},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()

    if "metaphor" in raw:
        return "metaphorical"
    if "literal" in raw:
        return "literal"
    return "unknown"


#Stats

def bootstrap_ci(vals: list[float], n_boot: int = 1000) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals)
    boots = [np.mean(np.random.choice(arr, len(arr), replace=True))
             for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def fmt(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    mu = np.mean(vals)
    lo, hi = bootstrap_ci(vals)
    return f"{mu:.3f} [{lo:.3f}, {hi:.3f}]  n={len(vals)}" #low, mean, high under bootstrap


# Per-file judging (the generation experiment produces one file per model per alpha)

def judge_file(fpath: str, model, tokenizer) -> dict:
    with open(fpath) as f:
        d = json.load(f)

    model_name = d["model"]
    alpha      = d["alpha"]
    steer_dir  = d["steer_dir"]
    examples   = d["examples"]

    print(f"\n{'='*60}")
    print(f"File      : {Path(fpath).name}")
    print(f"Model     : {model_name}")
    print(f"Alpha     : {alpha}  |  steer_dir: {steer_dir}  |  n={len(examples)}")
    print()

    base_correct, mm_flipped, das_flipped = [], [], []
    mm_degen, das_degen, base_degen = 0, 0, 0
    judged_examples = []

    for i, ex in enumerate(examples):
        adj          = ex["adj"]
        noun         = ex["noun"]
        true_label   = ex["true_label"]   # "literal" or "metaphorical"
        target_label = ex["steer_dir"]    # "metaphorical" or "literal"
        sentence     = f"The {noun} was {adj}."

        j_base = judge_completion(model, tokenizer, sentence, adj, ex["baseline"])
        j_mm   = judge_completion(model, tokenizer, sentence, adj, ex["mass_mean"])
        j_das  = judge_completion(model, tokenizer, sentence, adj, ex["das"])

        base_ok = int(j_base == true_label)   if j_base not in ("degenerate", "unknown") else None
        mm_ok   = int(j_mm   == target_label) if j_mm   not in ("degenerate", "unknown") else None
        das_ok  = int(j_das  == target_label) if j_das  not in ("degenerate", "unknown") else None

        if j_base == "degenerate": base_degen += 1
        if j_mm   == "degenerate": mm_degen   += 1
        if j_das  == "degenerate": das_degen  += 1

        if base_ok is not None: base_correct.append(base_ok)
        if mm_ok   is not None: mm_flipped.append(mm_ok)
        if das_ok  is not None: das_flipped.append(das_ok)

        def mark(ok, j):
            return "D" if j == "degenerate" else ("✓" if ok == 1 else ("✗" if ok == 0 else "?"))

        first_base = extract_first_sentence(ex["baseline"])
        first_mm   = extract_first_sentence(ex["mass_mean"])
        first_das  = extract_first_sentence(ex["das"])

        print(f"  [{i+1:>3}/{len(examples)}] {adj}+{noun}  true={true_label} → target={target_label}")
        print(f"    base ({str(j_base):>12}) {mark(base_ok, j_base)}: {first_base[:80]}")
        print(f"    mm   ({str(j_mm):>12})   {mark(mm_ok,   j_mm)}:   {first_mm[:80]}")
        print(f"    das  ({str(j_das):>12})   {mark(das_ok,  j_das)}:   {first_das[:80]}", flush=True)

        judged_examples.append({
            **ex,
            "first_sentence_baseline":  first_base,
            "first_sentence_mass_mean": first_mm,
            "first_sentence_das":       first_das,
            "judge_baseline":           j_base,
            "judge_mass_mean":          j_mm,
            "judge_das":                j_das,
            "base_correct":             base_ok,
            "mm_flipped":               mm_ok,
            "das_flipped":              das_ok,
        })

    # Filtered metrics: exclude few-shot adjs, baseline-correct only, degenerate=0
    filt_mm, filt_das = [], []
    for ex in judged_examples:
        if ex["adj"] in FEWSHOT_ADJS:
            continue
        if ex["base_correct"] != 1:
            continue
        filt_mm.append( ex["mm_flipped"]  if ex["mm_flipped"]  is not None else 0)
        filt_das.append(ex["das_flipped"] if ex["das_flipped"] is not None else 0)

    n = len(examples)
    n_filt = len(filt_mm)
    print(f"\n  baseline agree : {fmt(base_correct)}  (degenerate: {base_degen}/{n})")
    print(f"  mass_mean flip : {fmt(mm_flipped)}   (degenerate: {mm_degen}/{n})")
    print(f"  das flip       : {fmt(das_flipped)}   (degenerate: {das_degen}/{n})")
    print(f"\n  --- filtered (few-shot excl., baseline-correct, degen=0, n={n_filt}) ---")
    print(f"  mass_mean flip : {fmt(filt_mm)}")
    print(f"  das flip       : {fmt(filt_das)}")

    return {
        "source_file":           Path(fpath).name,
        "model":                 model_name,
        "alpha":                 alpha,
        "steer_dir":             steer_dir,
        "n":                     n,
        "base_degenerate":       base_degen,
        "mm_degenerate":         mm_degen,
        "das_degenerate":        das_degen,
        "baseline_agree":        float(np.mean(base_correct)) if base_correct else None,
        "mass_mean_flip":        float(np.mean(mm_flipped))   if mm_flipped   else None,
        "das_flip":              float(np.mean(das_flipped))  if das_flipped  else None,
        "baseline_ci":           bootstrap_ci(base_correct),
        "mass_mean_ci":          bootstrap_ci(mm_flipped),
        "das_ci":                bootstrap_ci(das_flipped),
        "n_filtered":            n_filt,
        "mm_flip_filtered":      float(np.mean(filt_mm))  if filt_mm  else None,
        "das_flip_filtered":     float(np.mean(filt_das)) if filt_das else None,
        "mm_flip_filtered_ci":   bootstrap_ci(filt_mm),
        "das_flip_filtered_ci":  bootstrap_ci(filt_das),
        "examples":              judged_examples,
    }


# Main 

def main():
    os.makedirs(RESULTS_OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(RESULTS_GLOB))
    if not files:
        print(f"No files found matching: {RESULTS_GLOB}")
        return

    print(f"Judge model : {JUDGE_MODEL}")
    print(f"Files       : {len(files)}")
    print(f"Repeat threshold for degenerate: {REPEAT_THRESHOLD}")
    for f in files:
        print(f"  {Path(f).name}")
    print()

    model, tokenizer = load_judge(JUDGE_MODEL)

    all_results = []
    for fpath in files:
        result = judge_file(fpath, model, tokenizer)
        all_results.append(result)
        # Checkpoint after every file
        out = Path(RESULTS_OUT_DIR) / f"judge_{Path(fpath).stem}.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  [saved] {out}", flush=True)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"SUMMARY — judge: {JUDGE_MODEL}")
    print(f"{'File':<50} {'dir':>5} {'α':>4} {'base':>7} {'mm(raw)':>9} {'das(raw)':>9} {'mm(filt)':>10} {'das(filt)':>10} {'n_filt':>7}")
    print("-" * 110)
    for r in all_results:
        name  = r["source_file"].replace("results_metaphor_generation_final_", "")[:49]
        ba    = f"{r['baseline_agree']:.3f}"     if r["baseline_agree"]    is not None else "n/a"
        mm    = f"{r['mass_mean_flip']:.3f}"     if r["mass_mean_flip"]    is not None else "n/a"
        da    = f"{r['das_flip']:.3f}"           if r["das_flip"]          is not None else "n/a"
        mm_f  = f"{r['mm_flip_filtered']:.3f}"  if r["mm_flip_filtered"]  is not None else "n/a"
        das_f = f"{r['das_flip_filtered']:.3f}" if r["das_flip_filtered"] is not None else "n/a"
        print(f"{name:<50} {r['steer_dir']:>5} {str(r['alpha']):>4} {ba:>7} {mm:>9} {da:>9} {mm_f:>10} {das_f:>10} {r['n_filtered']:>7}")

    summary_path = Path(RESULTS_OUT_DIR) / "judge_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"judge_model": JUDGE_MODEL, "results": all_results}, f, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
