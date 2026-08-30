"""Synthetic style-classification dataset for the QNLP track (Track B).

Why this exists: lambeq's DisCoCat pipeline needs a CCG parser to turn a
sentence into a string diagram. As of this writing, lambeq's default parsers
are broken at the infrastructure level — BobcatParser's model host
(qnlp.cambridgequantum.com) doesn't resolve, and WebParser's fallback API
(cqc.pythonanywhere.com) returns 404. See
https://github.com/Quantinuum/lambeq/issues/253 (open since Dec 2025, no fix).

Rather than depend on that parser, this dataset is generated as structured
(subject, verb, object) triples directly, so src/data/qnlp_diagrams.py can
hand-build the pregroup grammar diagram for each row without parsing
anything — the grammar structure (strict SVO) is fixed by construction, not
inferred.

--- Why this is a STYLE task, not a fact-checking task (a real pivot) ---

An earlier version of this dataset used arbitrary true/false facts (e.g.
"France contains Paris" vs. "Germany contains Paris"). That failed hard:
train loss went to 0 (perfect memorization) while validation accuracy sat
*below* chance, and neither more data (48->150 rows) nor fixing a real
vocabulary-coverage bug nor adding more circuit layers helped — every lever
that normally fixes underfitting/overfitting made it worse instead.

The diagnosis: DisCoCat's mechanism is that meaning *composes* — in
lambeq's own MC tutorial dataset, "chef"+"cooks"+"meal" compositionally
signals "food", and that signal transfers to new combinations because the
words share a semantic category. Arbitrary facts have no such structure:
nothing about the *meaning* of "France" and "Paris" compositionally encodes
their factual relationship, so a model with independent per-word parameters
just memorizes each word's role in its own few training sentences with zero
pressure to generalize.

This version instead classifies *rhetorical style* — credible-sounding vs.
sensational-sounding framing, which genuinely is compositional (the words
themselves carry the stylistic signal) and is a real fake-news feature
(the previous classical project used a "Sensationalism Index" as one of
its 15 linguistic features). A shared pool of subjects is crossed with two
style-coded (verb, object) pools, so every word is reused across many
distinct sentences (each subject appears 16 times, each verb/object 10
times) — the reuse that was structurally missing before.
"""

import random

SUBJECTS = [
    "Scientists", "Officials", "Researchers", "Analysts", "Doctors",
    "Experts", "Authorities", "Economists", "Regulators", "Auditors",
]

CREDIBLE_PAIRS = [
    ("confirm", "results"), ("report", "findings"), ("announce", "data"),
    ("verify", "evidence"), ("present", "conclusions"), ("publish", "statistics"),
    ("review", "records"), ("validate", "figures"),
]

SENSATIONAL_PAIRS = [
    ("expose", "secrets"), ("reveal", "scandals"), ("uncover", "conspiracies"),
    ("leak", "coverups"), ("unveil", "schemes"), ("disclose", "plots"),
    ("hint", "rumors"), ("spread", "gossip"),
]


def generate_claims(seed: int = 42) -> list[dict]:
    """Returns rows of {"subject", "verb", "object", "label", "text"}.
    label 0 = credible-style, 1 = sensational-style."""
    rng = random.Random(seed)
    rows = []
    for subject in SUBJECTS:
        for verb, obj in CREDIBLE_PAIRS:
            rows.append({"subject": subject, "verb": verb, "object": obj, "label": 0,
                         "text": f"{subject} {verb} {obj}."})
        for verb, obj in SENSATIONAL_PAIRS:
            rows.append({"subject": subject, "verb": verb, "object": obj, "label": 1,
                         "text": f"{subject} {verb} {obj}."})
    rng.shuffle(rows)
    return rows


def train_valid_test_split(rows: list[dict], train_frac: float = 0.7, valid_frac: float = 0.15):
    """Guarantees every word appears at least once in train.

    DisCoCat gives each word its own trainable parameters — there's no
    shared embedding space, so a word never seen in training keeps its
    random initial parameters and cannot be predicted correctly. This
    forces any row that introduces a not-yet-covered word into train
    first, then fills the rest of train/valid/test from what's left — so
    valid/test only ever test new *combinations* of known words, never
    entirely unknown words.
    """
    covered_vocab = set()
    forced_train = []
    remaining = []
    for row in rows:
        words = {row["subject"], row["verb"], row["object"]}
        if not words.issubset(covered_vocab):
            forced_train.append(row)
            covered_vocab.update(words)
        else:
            remaining.append(row)

    n = len(rows)
    n_train_target = int(n * train_frac)
    n_valid_target = int(n * valid_frac)

    extra_train_needed = max(0, n_train_target - len(forced_train))
    train = forced_train + remaining[:extra_train_needed]
    rest = remaining[extra_train_needed:]

    return {
        "train": train,
        "valid": rest[:n_valid_target],
        "test": rest[n_valid_target:],
    }


if __name__ == "__main__":
    claims = generate_claims()
    splits = train_valid_test_split(claims)
    vocab = set()
    for c in claims:
        vocab.update([c["subject"], c["verb"], c["object"]])
    print(f"Generated {len(claims)} claims (balanced: "
          f"{sum(1 for c in claims if c['label'] == 0)} credible / "
          f"{sum(1 for c in claims if c['label'] == 1)} sensational), "
          f"vocab size: {len(vocab)}")
    for name, split in splits.items():
        print(f"  {name}: {len(split)} rows")
    print("\nSample rows:")
    for row in claims[:5]:
        print(f"  [{'CREDIBLE' if row['label'] == 0 else 'SENSATIONAL'}] {row['text']}")
