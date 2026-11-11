import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.decomposition import PCA
from transformers import BitsAndBytesConfig
import nnsight


def rindex(seq, val):
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] == val:
            return i
    return -1


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


def normalize_phrase_no_period(s):
    s = str(s).strip()
    s = s.rstrip(".")
    return s


def get_template_task2(dataset):
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


def collect_train_activations(model, train_df, dataset, layer):
    PROMPT_TEMPLATE_M = get_template_M(dataset)

    activations = []

    punctuation_token_id = model.tokenizer(
        ".",
        add_special_tokens=False
    ).input_ids[-1]

    for i in trange(len(train_df)):
        row = train_df.iloc[i]

        prompt = PROMPT_TEMPLATE_M.format(statement=row["sentence"])

        prompt_token_ids = model.tokenizer(
            prompt,
            add_special_tokens=False
        ).input_ids

        dot_index = rindex(prompt_token_ids, punctuation_token_id)

        with torch.no_grad():
            with model.trace(prompt):
                activation = model.model.layers[layer].output[0][dot_index, :].save()
                activations.append(activation.unsqueeze(0))

    train_activations = torch.cat(activations)

    train_activations = (
        train_activations
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )
    print("train_df size:", len(train_df))
    print("train_activations shape:", train_activations.shape)
    print("label counts:")
    print(train_df["label_id"].value_counts())
    return train_activations


def collect_train_activations_task2(model, train_df, dataset, layer):
    PROMPT_TEMPLATE = get_template_task2(dataset)

    activations = []

    punctuation_token_id = model.tokenizer(
        ".",
        add_special_tokens=False
    ).input_ids[-1]

    for i in trange(len(train_df)):
        row = train_df.iloc[i]

        metaphor = normalize_phrase_no_period(row["sentence1"])
        rephrase = normalize_phrase_no_period(row["sentence2"])

        prompt = PROMPT_TEMPLATE.format(
            METAPHOR=metaphor,
            REPHRASE=rephrase,
        )

        prompt_token_ids = model.tokenizer(
            prompt,
            add_special_tokens=False
        ).input_ids

        dot_index = rindex(prompt_token_ids, punctuation_token_id)

        with torch.no_grad():
            with model.trace(prompt):
                activation = model.model.layers[layer].output[0][dot_index, :].save()
                activations.append(activation.unsqueeze(0))

    train_activations = torch.cat(activations)

    train_activations = (
        train_activations
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )
    print("train_df size:", len(train_df))
    print("train_activations shape:", train_activations.shape)
    print("label counts:")
    print(train_df["label_id"].value_counts())
    return train_activations


def plot_pca(train_activations, train_df, dataset, layer, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)

    labels = train_df["label_id"].to_numpy()

    pca = PCA(n_components=2)
    points = pca.fit_transform(train_activations)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        points[labels == 0, 0],
        points[labels == 0, 1],
        label="literal",
        alpha=0.7,
    )

    ax.scatter(
        points[labels == 1, 0],
        points[labels == 1, 1],
        label="metaphoric",
        alpha=0.7,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA visualization | {dataset} | layer {layer}")
    ax.legend()

    safe_model_name = model_name.replace("/", "_")
    output_path = os.path.join(
        output_dir,
        f"pca_{dataset}_layer{layer}_{safe_model_name}.png"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_path)


def plot_pca_task2(train_activations, train_df, dataset, layer, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)

    labels = train_df["label_id"].to_numpy()

    pca = PCA(n_components=2)
    points = pca.fit_transform(train_activations)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        points[labels == 0, 0],
        points[labels == 0, 1],
        label="incorrect",
        alpha=0.7,
    )

    ax.scatter(
        points[labels == 1, 0],
        points[labels == 1, 1],
        label="correct",
        alpha=0.7,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA visualization | task2 | {dataset} | layer {layer}")
    ax.legend()

    safe_model_name = model_name.replace("/", "_")
    output_path = os.path.join(
        output_dir,
        f"pca_task2_{dataset}_layer{layer}_{safe_model_name}.png"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_path)

'''
def collect_train_activations_task2(model, train_df, dataset, layer):
    PROMPT_TEMPLATE = get_template_task2(dataset)

    activations = []

    punctuation_token_id = model.tokenizer(
        ".",
        add_special_tokens=False
    ).input_ids[-1]

    for i in trange(len(train_df)):
        row = train_df.iloc[i]

        metaphor = normalize_phrase_no_period(row["sentence1"])
        rephrase = normalize_phrase_no_period(row["sentence2"])

        prompt = PROMPT_TEMPLATE.format(
            METAPHOR=metaphor,
            REPHRASE=rephrase,
        )

        prompt_token_ids = model.tokenizer(
            prompt,
            add_special_tokens=False
        ).input_ids

        dot_index = rindex(prompt_token_ids, punctuation_token_id)

        with torch.no_grad():
            with model.trace(prompt):
                activation = model.model.layers[layer].output[0][dot_index, :].save()
                activations.append(activation.unsqueeze(0))

    train_activations = torch.cat(activations)

    train_activations = (
        train_activations
        .detach()
        .to(dtype=torch.float32)
        .cpu()
        .numpy()
    )
    print("train_df size:", len(train_df))
    print("train_activations shape:", train_activations.shape)
    print("label counts:")
    print(train_df["label_id"].value_counts())
    return train_activations


def plot_pca(train_activations, train_df, dataset, layer, output_dir, model_name):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    os.makedirs(output_dir, exist_ok=True)

    labels = train_df["label_id"].to_numpy()

    pca = PCA(n_components=5)
    points = pca.fit_transform(train_activations)

    safe_model_name = model_name.replace("/", "_")

    fig, axes = plt.subplots(5, 5, figsize=(20, 20))

    for i in range(5):
        for j in range(5):
            ax = axes[i, j]

            ax.scatter(
                points[labels == 0, j],
                points[labels == 0, i],
                label="literal" if i == 0 and j == 0 else None,
                alpha=0.7,
                s=18,
            )

            ax.scatter(
                points[labels == 1, j],
                points[labels == 1, i],
                label="metaphoric" if i == 0 and j == 0 else None,
                alpha=0.7,
                s=18,
            )

            ax.set_xlabel(f"PC{j + 1}")
            ax.set_ylabel(f"PC{i + 1}")

            if i == j:
                ax.set_title(f"PC{i + 1} x PC{j + 1}")

    fig.suptitle(
        f"PCA first 5 PCs | {dataset} | layer {layer} | {model_name}",
        fontsize=18,
    )

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 0.97, 0.97])

    output_path = os.path.join(
        output_dir,
        f"pca_5x5_{dataset}_layer{layer}_{safe_model_name}.png"
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", output_path)
'''

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["task1", "task2"], default="task1")
    parser.add_argument("--dataset", type=str, choices=["adj", "noun", "verb"], required=True)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--output_dir", type=str, default="pca_outputs")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")

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
        from setup_load import load_dataset_task1

        df, df_new, train_df, test_df = load_dataset_task1(args.dataset)

        train_activations = collect_train_activations(
            model=model,
            train_df=train_df,
            dataset=args.dataset,
            layer=args.layer,
        )

        plot_pca(
            train_activations=train_activations,
            train_df=train_df,
            dataset=args.dataset,
            layer=args.layer,
            output_dir=args.output_dir,
            model_name=args.model_name

        )

    elif args.task == "task2":
        if args.dataset == "adj":
            raise ValueError("task2 currently supports only noun and verb datasets")

        from setup_load import load_dataset_task2

        df, df_new, train_df, test_df = load_dataset_task2(args.dataset)

        train_activations = collect_train_activations_task2(
            model=model,
            train_df=train_df,
            dataset=args.dataset,
            layer=args.layer,
        )

        plot_pca_task2(
            train_activations=train_activations,
            train_df=train_df,
            dataset=args.dataset,
            layer=args.layer,
            output_dir=args.output_dir,
            model_name=args.model_name

        )


if __name__ == "__main__":
    main()