import torch

from papers.p01_vilnis_2015.src.corpus import tokenize, build_vocab, make_skipgram_pairs, make_training_triples, sample_negative


def test_tokenize_lowercases_and_splits():
    assert tokenize("King Queen") == ["king", "queen"]


def test_build_vocab_creates_ids():
    tokens = ["king", "queen", "king"]

    word_to_id, id_to_word, counts = build_vocab(tokens)

    assert counts["king"] == 2
    # id_to_word maps ids back to tokens
    assert id_to_word[word_to_id["king"]] == "king"


def test_make_skipgram_pairs_window_one():
    tokens = ["a", "b", "c"]
    word_to_id, _, _ = build_vocab(tokens)

    pairs = make_skipgram_pairs(tokens, word_to_id, window_size=1)

    decoded = [
        (tokens[i], tokens[j])
        for i, j in pairs
    ]

    assert decoded == [
        ("a", "b"),
        ("b", "a"),
        ("b", "c"),
        ("c", "b"),
    ]


def test_sample_negative_is_not_positive():
    for _ in range(100):
        negative = sample_negative(context_id=1, vocab_size=5)
        assert negative != 1


def test_make_training_triples_shape():
    pairs = [(0, 1), (1, 2)]
    triples = make_training_triples(pairs, vocab_size=4)

    assert len(triples) == 2

    for center_id, positive_id, negative_id in triples:
        assert isinstance(center_id, int)
        assert isinstance(positive_id, int)
        assert isinstance(negative_id, int)
        assert negative_id != positive_id