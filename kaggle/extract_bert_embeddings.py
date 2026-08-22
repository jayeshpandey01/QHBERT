"""Run this as a Kaggle Script (or paste into a Kaggle Notebook) on GPU.

Nothing gets downloaded manually anywhere: the dataset is attached via
Kaggle's "+ Add Input" panel (mounts read-only under /kaggle/input/...),
transformers/torch are preinstalled on Kaggle, and the output .pt files
land in /kaggle/working/ where you download them from the notebook's
Output tab when done — that's the only file that ever leaves Kaggle.

Steps to use:
  1. New Notebook/Script on kaggle.com, enable GPU (Settings > Accelerator > GPU T4 x2).
  2. "+ Add Input" -> search "LIAR dataset" (or ISOT / WELFake) -> add it.
  3. Check the exact path Kaggle mounted it at (shown in the Input panel,
     usually /kaggle/input/<dataset-slug>/...) and set DATA_DIR below.
  4. Run all. Download /kaggle/working/liar_embeddings.pt when finished.
  5. Bring that single file back to your local machine and point
     src/training/train.py at it — no raw dataset ever touches local disk.
"""

import os

import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizerFast

# --- adjust after checking the Input panel on Kaggle ---
DATA_DIR = "/kaggle/input/liar-dataset"
OUTPUT_PATH = "/kaggle/working/liar_embeddings.pt"
BATCH_SIZE = 32
MAX_LENGTH = 512

# LIAR's raw TSVs have no header; this is the original 14-column schema.
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]
LIAR_LABEL_MAP = {
    "true": 0, "mostly-true": 0, "half-true": 0,
    "barely-true": 1, "false": 1, "pants-fire": 1,
}


def clean_text(text: str) -> str:
    import re
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_liar_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
    df = df.dropna(subset=["statement", "label"])
    df["label_binary"] = df["label"].map(LIAR_LABEL_MAP)
    df = df.dropna(subset=["label_binary"])
    df["text"] = df["statement"].apply(clean_text)
    return df


@torch.no_grad()
def extract_embeddings(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    all_embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=MAX_LENGTH).to(device)
        out = model(**encoded)
        cls_embeddings = out.last_hidden_state[:, 0, :].cpu()
        all_embeddings.append(cls_embeddings)
        if (i // batch_size) % 20 == 0:
            print(f"  {i}/{len(texts)} embedded")
    return torch.cat(all_embeddings, dim=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

    splits = {}
    for split in ["train", "test", "valid"]:
        candidates = [
            os.path.join(DATA_DIR, f"{split}.tsv"),
            os.path.join(DATA_DIR, f"{split}2.tsv"),
        ]
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path is None:
            print(f"Skipping {split}: no file found among {candidates}")
            continue

        print(f"Loading {split} from {path}")
        df = load_liar_split(path)
        print(f"  {len(df)} rows after cleaning/label mapping")

        embeddings = extract_embeddings(df["text"].tolist(), tokenizer, model, device)
        labels = torch.tensor(df["label_binary"].values, dtype=torch.long)
        splits[split] = {"embeddings": embeddings, "labels": labels}

    torch.save(splits, OUTPUT_PATH)
    print(f"Saved cached embeddings to {OUTPUT_PATH}")
    for split, data in splits.items():
        print(f"  {split}: embeddings {tuple(data['embeddings'].shape)}, labels {tuple(data['labels'].shape)}")


if __name__ == "__main__":
    main()
