"""Hand-built pregroup grammar diagrams for Subject-Verb-Object sentences.

Bypasses lambeq's automatic CCG parser entirely (see qnlp_synthetic_claims.py
for why it's currently broken). Since every row in the synthetic dataset is
strict SVO by construction, one diagram shape covers the whole dataset:

    Subject(n)  Verb(n.r @ s @ n.l)  Object(n)
         \\_________/                    |
           Cup(n, n.r)                   |
                \\____________________Cup(n.l, n)
                          -> s

This is the standard DisCoCat transitive-verb reduction: the verb's type
n.r @ s @ n.l "consumes" the subject on its left and the object on its
right via grammar cups, leaving a bare sentence wire `s`. Verified once
here rather than trusted blindly — see the __main__ block and
tests/test_qnlp_diagrams.py.
"""

from lambeq.backend.grammar import Cup, Id, Ty, Word

N = Ty("n")
S = Ty("s")


def svo_diagram(subject: str, verb: str, obj: str):
    """Builds the pregroup diagram for "<subject> <verb> <object>"."""
    subj_word = Word(subject, N)
    verb_word = Word(verb, N.r @ S @ N.l)
    obj_word = Word(obj, N)

    words = subj_word @ verb_word @ obj_word
    reduction = Cup(N, N.r) @ Id(S) @ Cup(N.l, N)
    return words >> reduction


if __name__ == "__main__":
    diagram = svo_diagram("France", "contains", "Paris")
    print(f"Diagram built: {len(diagram.boxes)} boxes")
    print(f"Domain: {diagram.dom}, Codomain: {diagram.cod}")
    assert diagram.cod == S, f"Expected codomain 's', got {diagram.cod}"
    print("OK — diagram reduces to a bare sentence type 's' as expected.")
