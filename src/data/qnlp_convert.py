"""Convert a "normal" text dataset into a QNLP-ready one.

"Quantum-ready" here means concretely: every remaining row is a sentence
that lambeq's CCG parser (Bobcat) can successfully reduce to a valid
DisCoCat string diagram, which can then be functorially mapped to a
quantum circuit. Real-world sentences are not guaranteed to parse — Bobcat
is documented to fail on real headlines with named entities and complex
phrasing (e.g. "Senate Advances Bill To Approve Keystone Pipeline Despite
Obama's Veto Threat"). Rather than assume it works, this module measures
the parse success rate directly and keeps only what actually parses.

This is source-agnostic on purpose: point any loader's (text, label) rows
at `make_quantum_ready`, whether that's LIAR, FEVER, or something else.
See load_liar() / load_fever() below for two ready-made sources — FEVER
is recommended first since its claims are short (avg. 9.8 tokens) and
built from simple Wikipedia-sentence mutations, so it should parse far
more cleanly than LIAR's political rhetoric.

Usage:
  .venv/Scripts/python.exe -m src.data.qnlp_convert --source fever --output data/fever_qnlp
  .venv/Scripts/python.exe -m src.data.qnlp_convert --source liar --input path/to/train.tsv --output data/liar_qnlp_train
"""

import argparse
import json


def filter_parseable(texts: list[str]) -> tuple[list[int], list[str]]:
    """Returns (indices that parsed successfully, error messages for the rest)."""
    from lambeq import BobcatParser

    parser = BobcatParser(verbose="suppress")
    parseable_indices = []
    errors = []
    for i, text in enumerate(texts):
        try:
            diagram = parser.sentence2diagram(text)
            if diagram is not None:
                parseable_indices.append(i)
            else:
                errors.append(f"[{i}] parser returned None: {text!r}")
        except Exception as e:
            errors.append(f"[{i}] {type(e).__name__}: {text!r} -> {e}")
    return parseable_indices, errors


def make_quantum_ready(rows: list[dict], output_path: str, source_name: str = "dataset") -> dict:
    """rows: list of {"text": str, "label": int}. Writes the parseable subset to
    <output_path>.jsonl and returns a summary dict (also useful directly in the paper)."""
    texts = [r["text"] for r in rows]
    parseable_indices, errors = filter_parseable(texts)
    success_rate = len(parseable_indices) / len(rows) if rows else 0.0

    print(f"[{source_name}] parse success rate: {success_rate:.2%} "
          f"({len(parseable_indices)}/{len(rows)})")
    if errors:
        print(f"[{source_name}] first 10 of {len(errors)} parse failures:")
        for err in errors[:10]:
            print(f"  {err}")

    out_file = f"{output_path}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for i in parseable_indices:
            f.write(json.dumps(rows[i]) + "\n")

    print(f"[{source_name}] wrote {len(parseable_indices)} quantum-ready rows to {out_file}")
    return {"source": source_name, "total": len(rows), "parseable": len(parseable_indices),
            "success_rate": success_rate, "output": out_file}


# --- source loaders: each returns list[{"text": str, "label": int}], label 0=true/real, 1=false/fake ---

LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]
LIAR_LABEL_MAP = {
    "true": 0, "mostly-true": 0, "half-true": 0,
    "barely-true": 1, "false": 1, "pants-fire": 1,
}


def load_liar(tsv_path: str) -> list[dict]:
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t", header=None, names=LIAR_COLUMNS)
    df = df.dropna(subset=["statement", "label"])
    df["label_binary"] = df["label"].map(LIAR_LABEL_MAP)
    df = df.dropna(subset=["label_binary"])
    return [{"text": row["statement"], "label": int(row["label_binary"])}
            for _, row in df.iterrows()]


def load_fever(split: str = "train", limit: int | None = None) -> list[dict]:
    """Pulls FEVER claims from HuggingFace (SUPPORTS -> 0, REFUTES -> 1, drops
    NOT ENOUGH INFO since it isn't a true/false label). Needs `datasets` installed.
    Run this on Kaggle (internet-enabled) to avoid any local download."""
    from datasets import load_dataset

    ds = load_dataset("fever", "v1.0", split=split, trust_remote_code=True)
    label_map = {"SUPPORTS": 0, "REFUTES": 1}
    rows = []
    for example in ds:
        if example["label"] not in label_map:
            continue
        rows.append({"text": example["claim"], "label": label_map[example["label"]]})
        if limit and len(rows) >= limit:
            break
    return rows


LOADERS = {"liar": load_liar, "fever": load_fever}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=list(LOADERS.keys()))
    parser.add_argument("--input", help="Path required for file-based sources (e.g. liar)")
    parser.add_argument("--output", required=True, help="Output path prefix (no extension)")
    parser.add_argument("--limit", type=int, default=None, help="Cap rows (useful for a quick trial run)")
    args = parser.parse_args()

    if args.source == "liar":
        if not args.input:
            raise SystemExit("--input is required for --source liar")
        rows = load_liar(args.input)
    else:
        rows = load_fever(limit=args.limit)

    if args.limit:
        rows = rows[:args.limit]

    make_quantum_ready(rows, args.output, source_name=args.source)


if __name__ == "__main__":
    main()
