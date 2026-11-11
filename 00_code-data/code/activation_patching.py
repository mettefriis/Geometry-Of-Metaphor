import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from transformers import BitsAndBytesConfig
import nnsight
import matplotlib.pyplot as plt



def get_template_M(dataset):
    if dataset == "adj":
        return """The town is sleepy. This sentence is: metaphoric
The sand is white. This sentence is: literal
The moon is young. This sentence is: metaphoric
The banana is ripe. This sentence is: literal
{statement} This sentence is:"""

    if dataset == "noun":
        return """The politician is glass. This sentence is: metaphoric
The politician is transparent. This sentence is: literal
The cake is heaven. This sentence is: metaphoric
The cake is delicious. This sentence is: literal
{statement} This sentence is:"""

    if dataset == "verb":
        return """Exercise drained her. This sentence is: metaphoric
Exercise depleted her. This sentence is: literal
The dancer floated. This sentence is: metaphoric
The dancer moved. This sentence is: literal
{statement} This sentence is:"""

    raise ValueError("dataset must be one of: adj, noun, verb")


TASKS = {
    "adj": [
        {
            "name": "adj_dry",
            "lit": "The skin is dry.",
            "met": "The wit is dry.",
        },
    
    ],
 
    "noun": [
        #{
         #   "name": "noun_lava",
          #  "lit": "The soup is hot.",
           # "met": "The soup is lava.",
        #},
        
        #{
         #   "name": "noun_engine",
          #  "lit": "The car has a small engine.",
           # "met": "The car has a toy engine.",
        #},
        {
            "name": "noun_breeze",
            "lit": "The breeze is cold.",
            "met": "The breeze is ice.",
        },
    ],
}

def rindex(seq, val):
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] == val:
            return i
    return -1


def load_model(model_name, load_in_4bit=False, load_in_8bit=False):
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        quantization_config = None

    model = nnsight.LanguageModel(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        dispatch=True,
    )

    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    return model


def get_label_token_ids(model):
    meta_ids = model.tokenizer(
        " metaph",
        add_special_tokens=False
    ).input_ids

    lit_ids = model.tokenizer(
        " literal",
        add_special_tokens=False
    ).input_ids

    metaphoric_token_id = meta_ids[0]
    literal_token_id = lit_ids[0]

    print("metaphoric token ids:", meta_ids)
    print("literal token ids:", lit_ids)
    print("metaphoric first token:", model.tokenizer.decode([metaphoric_token_id]))
    print("literal first token:", model.tokenizer.decode([literal_token_id]))

    return metaphoric_token_id, literal_token_id

def get_task2_template(dataset):
    if dataset == "noun":
        return (
            "Determine whether the paraphrase is apt or inapt.\n"
            "The politician is glass: The politician is transparent. This paraphrase is: apt.\n"
            "The politician is a wall: The politician is transparent. This paraphrase is: inapt.\n"
            "The cake is heaven: The cake is not delicious. This paraphrase is: inapt.\n"
            "The cake is heaven: The cake is delicious. This paraphrase is: apt.\n"
            "{METAPHOR}: {REPHRASE}. This paraphrase is:"
        )

    if dataset == "verb":
        return (
            "Determine whether the paraphrase is apt or inapt.\n"
            "Exercise drained her: Exercise depleted her. This paraphrase is: apt.\n"
            "Exercise drained her: Exercise drank her. This paraphrase is: inapt.\n"
            "The dancer floated: The dancer drifted. This paraphrase is: inapt.\n"
            "The dancer floated: The dancer moved. This paraphrase is: apt.\n"
            "{METAPHOR}: {REPHRASE}. This paraphrase is:"
        )

    raise ValueError("task2 dataset must be one of: noun, verb")

TASK2_TASKS = {
    "noun": [
        {
            "name": "task2_noun_puzzle_rocket_science",
            "metaphor": "The puzzle is rocket science",
            "rephrase": "The puzzle is difficult",
            "incor": "The puzzle is easy",
        },
    ]
}

def normalize_no_period(s):
    return str(s).strip().rstrip(".")


def get_task2_label_token_ids(model):
    apt_ids = model.tokenizer(
        " apt",
        add_special_tokens=False
    ).input_ids

    inapt_ids = model.tokenizer(
        " in",
        add_special_tokens=False
    ).input_ids

    apt_token_id = apt_ids[0]
    inapt_token_id = inapt_ids[0]

    print("apt token ids:", apt_ids)
    print("in token ids:", inapt_ids)
    print("apt chosen token:", model.tokenizer.decode([apt_token_id]))
    print("in chosen token:", model.tokenizer.decode([inapt_token_id]))

    return apt_token_id, inapt_token_id


def run_activation_patching_task2(model, dataset, pair_index, output_dir):
    """
    Task2 activation patching.

    This intentionally mirrors run_activation_patching() for task1:
      1. build source prompt and base prompt
      2. tokenize both prompts
      3. find the final prompt line using the last newline token
      4. patch source -> base token-by-token across that whole final-line span
      5. make the same layer x token heatmap

    Task2-specific changes only:
      - source = METAPHOR: REPHRASE. This paraphrase is:
      - base   = METAPHOR: INCOR.    This paraphrase is:
      - score  = P(" apt") - P(" in")
    """
    if dataset not in TASK2_TASKS:
        raise ValueError("task2 dataset must be one of: noun, verb")

    PROMPT_TEMPLATE_TASK2 = get_task2_template(dataset)
    pair = TASK2_TASKS[dataset][pair_index]
    metaphor = normalize_no_period(pair["metaphor"])
    rephrase = normalize_no_period(pair["rephrase"])
    incor = normalize_no_period(pair["incor"])
    pair_name = pair["name"]

    source_prompt = PROMPT_TEMPLATE_TASK2.format(
        METAPHOR=metaphor,
        REPHRASE=rephrase,
    )
    base_prompt = PROMPT_TEMPLATE_TASK2.format(
        METAPHOR=metaphor,
        REPHRASE=incor,
    )

    print(f"\n===== Running task2 {dataset} =====")
    print("Source apt:", f"{metaphor}: {rephrase}.")
    print("Base inapt:", f"{metaphor}: {incor}.")

    apt_token_id, in_token_id = get_task2_label_token_ids(model)

    # Keep exactly the same span logic as task1: start after the last newline.
    '''
    source_ids = model.tokenizer(source_prompt).input_ids
    base_ids = model.tokenizer(base_prompt).input_ids
    newline_id = model.tokenizer("\n").input_ids[-1]

    src_last_nl = rindex(source_ids, newline_id)
    base_last_nl = rindex(base_ids, newline_id)

    src_start = src_last_nl + 1 if src_last_nl != -1 else 0
    base_start = base_last_nl + 1 if base_last_nl != -1 else 0
    '''
    source_ids = model.tokenizer(source_prompt).input_ids
    base_ids = model.tokenizer(base_prompt).input_ids

    source_prefix = source_prompt.rsplit("\n", 1)[0] + "\n"
    base_prefix = base_prompt.rsplit("\n", 1)[0] + "\n"

    src_start = len(model.tokenizer(source_prefix).input_ids)
    base_start = len(model.tokenizer(base_prefix).input_ids)
    src_span_len = len(source_ids) - src_start
    base_span_len = len(base_ids) - base_start
    span_len = min(src_span_len, base_span_len)
    print("src_start:", src_start)
    print("base_start:", base_start)
    print("source span len:", src_span_len)
    print("base span len:", base_span_len)
    print("used span len:", span_len)

    source_activations = {}

    with torch.no_grad():
        with model.trace(source_prompt):
            for layer_index, layer in enumerate(model.model.layers):
                source_activations[layer_index] = layer.output[0].save()

    print("Collected source activations:", len(source_activations))
    print("Number of layers:", model.config.num_hidden_layers)

    if len(source_activations) != model.config.num_hidden_layers:
        raise RuntimeError(
            f"Expected {model.config.num_hidden_layers} activations, "
            f"got {len(source_activations)}"
        )

    patching_results = []

    # Keep exactly the same layer x token loop structure as task1.
    for layer_index in trange(model.config.num_hidden_layers):
        per_layer_diffs = []

        for offset in range(span_len):
            src_tok = src_start + offset
            base_tok = base_start + offset

            with torch.no_grad():
                with model.trace(base_prompt):
                    model.model.layers[layer_index].output[0][base_tok, :] = (
                        source_activations[layer_index][src_tok, :]
                    )
                    probs = model.output.logits[:, -1].softmax(dim=-1)

                    p_apt = probs[0, apt_token_id].item()
                    p_in = probs[0, in_token_id].item()

                    per_layer_diffs.append(p_apt - p_in)

        patching_results.append(per_layer_diffs)

    token_ids_span = base_ids[base_start: base_start + span_len]
    token_strings = [model.tokenizer.decode([tid]) for tid in token_ids_span]
    print("patched base token strings:", token_strings)

    data = np.array(patching_results)

    os.makedirs(output_dir, exist_ok=True)

    safe_model_name = model.config._name_or_path.replace("/", "_")
    output_path = os.path.join(
        output_dir,
        f"patching_task2_{pair_name}_{safe_model_name}.png"
    )


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    heat = ax.imshow(
        data,
        aspect="auto",
        cmap="Blues",
        interpolation="none",
    )

    LABEL_SIZE = 18
    TICK_SIZE = 14
    CBAR_SIZE = 14

    # remove x-axis title
    ax.set_ylabel(
        "Layer",
        fontsize=LABEL_SIZE
    )

    ax.set_xticks(range(len(token_strings)))

    ax.set_xticklabels(
        token_strings,
        rotation=45,
        ha="right",
        fontsize=TICK_SIZE
    )

    num_layers = data.shape[0]

    if num_layers <= 20:
        ax.set_yticks(range(num_layers))

        ax.set_yticklabels(
            range(num_layers),
            fontsize=TICK_SIZE
        )
    else:
        step = 5
        yticks = list(range(0, num_layers, step))

        ax.set_yticks(yticks)

        ax.set_yticklabels(
            yticks,
            fontsize=TICK_SIZE
        )

    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    cbar = fig.colorbar(
        heat,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(
        "P(metaphoric) − P(literal)",
        fontsize=CBAR_SIZE
    )

    cbar.ax.tick_params(labelsize=CBAR_SIZE)

    plt.tight_layout(pad=1.5)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
    print("Saved:", output_path)

    return data


def run_activation_patching(model, dataset, pair_index, output_dir):
    PROMPT_TEMPLATE_M = get_template_M(dataset)

    pair = TASKS[dataset][pair_index]

    source_statement_m = pair["met"]
    base_statement_lit = pair["lit"]
    pair_name = pair["name"]

    source_prompt_m = PROMPT_TEMPLATE_M.format(statement=source_statement_m)
    base_prompt_lit = PROMPT_TEMPLATE_M.format(statement=base_statement_lit)

    print(f"\n===== Running {dataset} =====")
    print("Source metaphoric:", source_statement_m)
    print("Base literal:", base_statement_lit)

    metaphoric_token_id, literal_token_id = get_label_token_ids(model)

    source_ids = model.tokenizer(source_prompt_m).input_ids
    base_ids = model.tokenizer(base_prompt_lit).input_ids
    newline_id = model.tokenizer("\n").input_ids[-1]

    src_last_nl = rindex(source_ids, newline_id)
    base_last_nl = rindex(base_ids, newline_id)

    src_start = src_last_nl + 1 if src_last_nl != -1 else 0
    base_start = base_last_nl + 1 if base_last_nl != -1 else 0

    src_span_len = len(source_ids) - src_start
    base_span_len = len(base_ids) - base_start
    span_len = min(src_span_len, base_span_len)

    print("source span len:", src_span_len)
    print("base span len:", base_span_len)
    print("used span len:", span_len)

    source_activations = {}

    with torch.no_grad():
        with model.trace(source_prompt_m):
            for layer_index, layer in enumerate(model.model.layers):
                source_activations[layer_index] = layer.output[0].save()

    print("Collected source activations:", len(source_activations))
    print("Collected source activations:", len(source_activations))
    print("Number of layers:", model.config.num_hidden_layers)

    if len(source_activations) != model.config.num_hidden_layers:
        raise RuntimeError(
            f"Expected {model.config.num_hidden_layers} activations, "
            f"got {len(source_activations)}"
        )
    patching_results = []

    for layer_index in trange(model.config.num_hidden_layers):
        per_layer_diffs = []

        for offset in range(span_len):
            src_tok = src_start + offset
            base_tok = base_start + offset

            with torch.no_grad():
                with model.trace(base_prompt_lit):
                    model.model.layers[layer_index].output[0][base_tok, :] = (
                        source_activations[layer_index][src_tok, :]
                    )
                    probs = model.output.logits[:, -1].softmax(dim=-1)

                    p_meta = probs[0, metaphoric_token_id].item()
                    p_lit = probs[0, literal_token_id].item()

                    per_layer_diffs.append(p_meta - p_lit)

        patching_results.append(per_layer_diffs)

    token_ids_span = base_ids[base_start: base_start + span_len]
    token_strings = [model.tokenizer.decode([tid]) for tid in token_ids_span]

    data = np.array(patching_results)

    os.makedirs(output_dir, exist_ok=True)

    safe_model_name = model.config._name_or_path.replace("/", "_")
    output_path = os.path.join(
        output_dir,
        f"patching_task1_{pair_name}_{safe_model_name}.png"
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    heat = ax.imshow(
        data,
        aspect="auto",
        cmap="Blues",
        interpolation="none",
    )

    LABEL_SIZE = 18
    TICK_SIZE = 14
    CBAR_SIZE = 14

    # remove x-axis title
    ax.set_ylabel(
        "Layer",
        fontsize=LABEL_SIZE
    )

    ax.set_xticks(range(len(token_strings)))

    ax.set_xticklabels(
        token_strings,
        rotation=45,
        ha="right",
        fontsize=TICK_SIZE
    )

    num_layers = data.shape[0]

    if num_layers <= 20:
        ax.set_yticks(range(num_layers))

        ax.set_yticklabels(
            range(num_layers),
            fontsize=TICK_SIZE
        )
    else:
        step = 5
        yticks = list(range(0, num_layers, step))

        ax.set_yticks(yticks)

        ax.set_yticklabels(
            yticks,
            fontsize=TICK_SIZE
        )

    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    cbar = fig.colorbar(
        heat,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(
        "P(metaphoric) − P(literal)",
        fontsize=CBAR_SIZE
    )

    cbar.ax.tick_params(labelsize=CBAR_SIZE)

    plt.tight_layout(pad=1.5)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print("Saved:", output_path)

    return data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["task1", "task2"],
        default="task1",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["adj", "noun", "verb", "all"],
        default="all",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="patching_outputs",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
    )

    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
    )

    args = parser.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one: --load_in_4bit or --load_in_8bit")
    print("CUDA available:", torch.cuda.is_available())

    model = load_model(
        args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    if args.task == "task1":
        if args.dataset == "all":
            datasets = ["adj", "noun"]
        else:
            datasets = [args.dataset]

        for dataset in datasets:
            for pair_index in range(len(TASKS[dataset])):
                run_activation_patching(
                    model=model,
                    dataset=dataset,
                    pair_index=pair_index,
                    output_dir=args.output_dir,
                )

    elif args.task == "task2":
        if args.dataset == "all":
            datasets = ["noun"]
        else:
            datasets = [args.dataset]

        for dataset in datasets:
            for pair_index in range(len(TASK2_TASKS[dataset])):
                run_activation_patching_task2(
                    model=model,
                    dataset=dataset,
                    pair_index=pair_index,
                    output_dir=args.output_dir,
                )

if __name__ == "__main__":
    main()