from transformers import logging
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
logging.set_verbosity_error()
import argparse
import subprocess
import sys
import os
import random
import torch
from sklearn.model_selection import train_test_split




def normalize_sentence(s):
    s = str(s).strip()
    s = s.rstrip(".")  
    return s + "."      


def normalize_phrase_no_period(s):
    return str(s).strip().rstrip(".").strip()

def setup(model_name):
    subprocess.run([sys.executable, "-m", "pip", "install", "nnsight>0.5.0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "transformers", "accelerate", "bitsandbytes"], check=True)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        login()

    random.seed(12)
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)

    # utility function
    def rindex(lst, value):
      """get the rightmost index of a value in a list."""
      return len(lst) - 1 - lst[::-1].index(value)

    class COLORS:
      """keep consistent plotting colors"""
      LIGHT_BLUE = "#46B1E1"
      BLUE = "#156082"
      LIGHT_ORANGE = "#F2AA84"
      ORANGE = "#E97132"
      PURPLE = "#A02B93"
      GREEN = "#4EA72E"

    def hex_to_rgba(hex_color, alpha=1.):
      """convert hex to rgb with opacity parameter"""
      hex_color = hex_color.lstrip('#')
      rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
      return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,

    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    return {
        #"quantization_config": quantization_config,
        "rindex": rindex,
        "COLORS": COLORS,
        "hex_to_rgba": hex_to_rgba,
        "model": model,
        "tokenizer": tokenizer,

    }


def load_adj():
    file1 = "/home/sli2/metaphor/data/an_mets.txt"   # metaphor
    file2 = "/home/sli2/metaphor/data/an_nonmets.txt"   # literal

    def process_txt(file_path):
        sentences = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                words = line.split()
                if len(words) < 2:
                    continue

                first_word, second_word = words[0], words[1]
                sentences.append(f"The {second_word} is {first_word}.")
        return sentences

    metaphor_list = process_txt(file1)
    literal_list = process_txt(file2)

    df = pd.DataFrame({
        "metaphoric": pd.Series(metaphor_list, dtype="string"),
        "literal": pd.Series(literal_list, dtype="string"),
    })


    df_met = df[['metaphoric']].rename(columns={'metaphoric': 'sentence'})
    df_met['label'] = 'metaphoric'
    df_met['label_id'] = 1

    df_lit = df[['literal']].rename(columns={'literal': 'sentence'})
    df_lit['label'] = 'literal'
    df_lit['label_id'] = 0

    df_new = pd.concat([df_met, df_lit], ignore_index=True)
    df_new = df_new[df_new['label'].isin(['literal', 'metaphoric'])].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df_new,          
        test_size=0.2,  
        shuffle=True,
        random_state=42 
    )

    print(len(train_df), len(test_df))

    print(train_df.head())
    print(test_df.head())

    return df, df_new, train_df, test_df


def load_noun():

    df = pd.read_csv('/home/sli2/metaphor/data/figqa_sim.txt')
    df_meta = df[['startphrase']].copy()
    df_meta['label'] = 'metaphoric'
    df_meta = df_meta.rename(columns={'startphrase': 'sentence'})
    df_lit_0 = df[df['labels'] == 0][['ending1']].copy()
    df_lit_0 = df_lit_0.rename(columns={'ending1': 'sentence'})
    df_lit_0['label'] = 'literal'
    df_lit_1 = df[df['labels'] == 1][['ending2']].copy()
    df_lit_1 = df_lit_1.rename(columns={'ending2': 'sentence'})
    df_lit_1['label'] = 'literal'
    df_lit = pd.concat([df_lit_0, df_lit_1], ignore_index=True)
    df_new = pd.concat([df_meta, df_lit], ignore_index=True)
    df_new["sentence"] = df_new["sentence"].apply(normalize_sentence)
    df_new['label_id'] = df_new['label'].map({
        'literal': 0,
        'metaphoric': 1
    })
    df_new = df_new.drop_duplicates(subset=['sentence'])

    train_df, test_df = train_test_split(
        df_new,          
        test_size=0.2,  
        shuffle=True,
        random_state=42 
    )
    print(train_df.head())
    print(len(train_df), len(test_df))

    return df, df_new, train_df, test_df


def load_verb():
    df = pd.read_csv('/home/sli2/metaphor/data/short-paraphrases-main.txt', sep='\t')
    df1 = df[['simple1 sentence', 'classt1']].rename(
        columns={'simple1 sentence': 'sentence', 'classt1': 'label'}
    )

    df2 = df[['simple2 sentence', 'classt2']].rename(
        columns={'simple2 sentence': 'sentence', 'classt2': 'label'}
    )

    df3 = df[['simple3 sentence', 'classt3']].rename(
        columns={'simple3 sentence': 'sentence', 'classt3': 'label'}
    )

    df_new = pd.concat([df1, df2, df3], ignore_index=True)
    df_new["sentence"] = df_new["sentence"].apply(normalize_sentence)
    df_new['label_id'] = df_new['label'].map({
        'literal': 0,
        'metaphoric': 1
    })
    df_new= df_new[df_new["label"] != "incorrect"]
    train_df, test_df = train_test_split(
        df_new,          
        test_size=0.2,  
        shuffle=True,
        random_state=42 
    )

    print(len(train_df), len(test_df))
    print(train_df.head())
    return df, df_new, train_df, test_df


def load_dataset_task1(dataset):
    if dataset == "adj":
        return load_adj()
    if dataset == "noun":
        return load_noun()
    if dataset == "verb":
        return load_verb()
    raise ValueError("dataset must be one of: adj, noun, verb")

def get_template_M(dataset):
    if dataset == "adj":
        PROMPT_TEMPLATE_M = """The town is sleepy. This sentence is: metaphoric
The sand is white. This sentence is: literal
The moon is young. This sentence is: metaphoric
The banana is ripe. This sentence is: literal
{statement} This sentence is:"""
        return PROMPT_TEMPLATE_M

    if dataset == "noun":
        PROMPT_TEMPLATE_M = """The politician is glass. This sentence is: metaphoric
The politician is transparent. This sentence is: literal
The cake is heaven. This sentence is: metaphoric
The cake is delicious. This sentence is: literal
{statement}. This sentence is:"""
        return PROMPT_TEMPLATE_M

    if dataset == "verb":
        PROMPT_TEMPLATE_M = """Exercise drained her. This sentence is: metaphoric
Exercise depleted her. This sentence is: literal
The dancer floated. This sentence is: metaphoric
The dancer moved. This sentence is: literal
{statement}. This sentence is:"""
        return PROMPT_TEMPLATE_M

    raise ValueError("dataset must be one of: adj, noun, verb")


def test_accuracy(model, tokenizer, train_df, test_df, dataset):
    from tqdm import trange

    PROMPT_TEMPLATE_M = get_template_M(dataset)
    eval_df = pd.concat([train_df, test_df], ignore_index=True)

    device = model.device

    literal_id = tokenizer.encode(" literal", add_special_tokens=False)[0]
    metaphoric_id = tokenizer.encode(" metaph", add_special_tokens=False)[0]

    predictions = []
    gold_labels = []

    with torch.no_grad():
        for i in trange(len(eval_df)):
            row = eval_df.iloc[i]
            statement = row["sentence"]

            if pd.isna(statement):
                continue

            prompt = PROMPT_TEMPLATE_M.format(statement=statement)

            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]

            literal_logit = logits[0, literal_id].item()
            metaphoric_logit = logits[0, metaphoric_id].item()

            pred_id = 1 if metaphoric_logit > literal_logit else 0

            predictions.append(pred_id)
            gold_labels.append(row["label_id"])

    correct = sum(p == l for p, l in zip(predictions, gold_labels))
    acc = correct / len(gold_labels)

    print("dataset =", dataset)
    print("size =", len(gold_labels))
    print("accuracy =", acc)

    return acc, predictions


# =========================
# Task 2: paraphrase aptness
# =========================

def load_noun_task2():
    """
    Task2 noun / FigQA-style sentence-pair dataset.

    Output columns:
      sentence1: metaphor
      sentence2: candidate paraphrase
      label:     correct / incorrect
      label_id:  correct=1, incorrect=0
    """

    df = pd.read_csv('/home/sli2/metaphor/data/figqa_sim.txt')

    df1 = pd.DataFrame({
        "sentence1": df["startphrase"],
        "sentence2": df["ending1"],
        "label": df["labels"].map({0: "correct", 1: "incorrect"}),
    })

    df2 = pd.DataFrame({
        "sentence1": df["startphrase"],
        "sentence2": df["ending2"],
        "label": df["labels"].map({0: "incorrect", 1: "correct"}),
    })

    df_new = pd.concat([df1, df2], ignore_index=True)
    df_new = df_new.dropna(subset=["sentence1", "sentence2", "label"]).reset_index(drop=True)

    df_new["sentence1"] = df_new["sentence1"].apply(normalize_phrase_no_period)
    df_new["sentence2"] = df_new["sentence2"].apply(normalize_phrase_no_period)
    df_new["label_id"] = df_new["label"].map({
        "correct": 1,
        "incorrect": 0,
    })

    train_df, test_df = train_test_split(
        df_new,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    print(train_df.head())
    print(len(train_df), len(test_df))

    return df, df_new, train_df, test_df


def load_verb_task2():
    """
    Task2 verb / short paraphrases sentence-pair dataset.

    Output columns:
      sentence1: metaphor/source sentence
      sentence2: candidate paraphrase
      label:     correct / incorrect
      label_id:  correct=1, incorrect=0
    """

    df = pd.read_csv('/home/sli2/metaphor/data/short-paraphrases-main.txt', sep='\t')

    pair_correct = pd.DataFrame({
        "sentence1": df["simple1 sentence"],
        "sentence2": df["simple2 sentence"],
        "label": "correct",
    })

    pair_incorrect = pd.DataFrame({
        "sentence1": df["simple1 sentence"],
        "sentence2": df["simple3 sentence"],
        "label": "incorrect",
    })

    df_new = pd.concat([pair_correct, pair_incorrect], ignore_index=True)
    df_new = df_new.dropna(subset=["sentence1", "sentence2", "label"]).reset_index(drop=True)

    df_new["sentence1"] = df_new["sentence1"].apply(normalize_phrase_no_period)
    df_new["sentence2"] = df_new["sentence2"].apply(normalize_phrase_no_period)
    df_new["label_id"] = df_new["label"].map({
        "correct": 1,
        "incorrect": 0,
    })

    train_df, test_df = train_test_split(
        df_new,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    print(train_df.head())
    print(len(train_df), len(test_df))

    return df, df_new, train_df, test_df


def load_dataset_task2(dataset):
    if dataset == "noun":
        return load_noun_task2()
    if dataset == "verb":
        return load_verb_task2()
    raise ValueError("task2 dataset must be one of: noun, verb")


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


def test_accuracy_task2(model, tokenizer, train_df, test_df, dataset):

    prompt_template = get_template_task2(dataset)
    eval_df = pd.concat([train_df, test_df], ignore_index=True)

    device = model.device

    # correct=apt=1, incorrect=inapt=0
    apt_id = tokenizer.encode(" apt", add_special_tokens=False)[0]
    inapt_id = tokenizer.encode(" inapt", add_special_tokens=False)[0]

    print("apt token id:", apt_id, "=>", tokenizer.decode([apt_id]))
    print("inapt token id:", inapt_id, "=>", tokenizer.decode([inapt_id]))

    predictions = []
    gold_labels = []

    with torch.no_grad():
        for i in trange(len(eval_df)):
            row = eval_df.iloc[i]
            sentence1 = row["sentence1"]
            sentence2 = row["sentence2"]

            if pd.isna(sentence1) or pd.isna(sentence2):
                continue

            metaphor = normalize_phrase_no_period(sentence1)
            rephrase = normalize_phrase_no_period(sentence2)

            prompt = prompt_template.format(
                METAPHOR=metaphor,
                REPHRASE=rephrase,
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]

            apt_logit = logits[0, apt_id].item()
            inapt_logit = logits[0, inapt_id].item()

            pred_id = 1 if apt_logit > inapt_logit else 0

            predictions.append(pred_id)
            gold_labels.append(int(row["label_id"]))

    correct = sum(p == l for p, l in zip(predictions, gold_labels))
    acc = correct / len(gold_labels)

    print("task = task2")
    print("dataset =", dataset)
    print("size =", len(gold_labels))
    print("accuracy =", acc)

    return acc, predictions


def load_dataset(task, dataset):
    if task == "task1":
        return load_dataset_task1(dataset)
    if task == "task2":
        return load_dataset_task2(dataset)
    raise ValueError("task must be one of: task1, task2")


def test_accuracy_by_task(model, tokenizer, train_df, test_df, dataset, task):
    if task == "task1":
        return test_accuracy(model, tokenizer, train_df, test_df, dataset)
    if task == "task2":
        return test_accuracy_task2(model, tokenizer, train_df, test_df, dataset)
    raise ValueError("task must be one of: task1, task2")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["task1", "task2"], default="task1")
    parser.add_argument("--dataset", choices=["adj", "noun", "verb"], required=True)
    parser.add_argument("--no_setup", action="store_true")
    parser.add_argument("--test_acc", action="store_true")
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()

    if args.task == "task2" and args.dataset == "adj":
        raise ValueError("task2 currently supports only noun and verb datasets")

    if not args.no_setup:
        setup_outputs = setup(args.model_name)
        model = setup_outputs["model"]
        tokenizer = setup_outputs["tokenizer"]

    df, df_new, train_df, test_df = load_dataset(args.task, args.dataset)

    if args.test_acc:
        if args.no_setup:
            raise ValueError("test_acc needs model, so do not use --no_setup")
        acc, preds = test_accuracy_by_task(
            model=model,
            tokenizer=tokenizer,
            train_df=train_df,
            test_df=test_df,
            dataset=args.dataset,
            task=args.task,
        )

if __name__ == "__main__":
    main()
