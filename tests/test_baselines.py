import torch

from src.models.baselines import BiLSTMClassifier, CNNClassifier, TransformerEncoderClassifier, build_tfidf_svm

VOCAB_SIZE = 100
SEQ_LEN = 20
BATCH = 4


def test_bilstm_forward_and_backward():
    model = BiLSTMClassifier(vocab_size=VOCAB_SIZE)
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits = model(x)
    assert logits.shape == (BATCH, 2)
    logits.sum().backward()


def test_cnn_forward_and_backward():
    model = CNNClassifier(vocab_size=VOCAB_SIZE)
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits = model(x)
    assert logits.shape == (BATCH, 2)
    logits.sum().backward()


def test_transformer_forward_and_backward():
    model = TransformerEncoderClassifier(vocab_size=VOCAB_SIZE, max_len=SEQ_LEN)
    x = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    logits = model(x)
    assert logits.shape == (BATCH, 2)
    logits.sum().backward()


def test_tfidf_svm_fits_and_predicts():
    texts = ["real news about the economy", "fake shocking secret exposed",
             "official government report released", "you wont believe this hoax"]
    labels = [0, 1, 0, 1]
    pipeline = build_tfidf_svm()
    pipeline.fit(texts, labels)
    preds = pipeline.predict(texts)
    assert len(preds) == len(labels)
