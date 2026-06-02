import random
from collections import Counter


def tokenize(text):
    """
    Minimal lowercase whitespace tokenizer.
    """
    return text.lower().split()


def build_vocab(tokens, min_count=1):
    """
    Build word-to-index and index-to-word mappings.
    """
    counts = Counter(tokens)

    words = [
        word
        for word, count in counts.items()
        if count >= min_count
    ]

    words = sorted(words)

    word_to_id = {word: idx for idx, word in enumerate(words)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}

    return word_to_id, id_to_word, counts


def make_skipgram_pairs(tokens, word_to_id, window_size=2):
    """
    Create center-context pairs from a token sequence.
    """
    pairs = []

    for center_pos, center_word in enumerate(tokens):
        if center_word not in word_to_id:
            continue

        center_id = word_to_id[center_word]

        left = max(0, center_pos - window_size)
        right = min(len(tokens), center_pos + window_size + 1)

        for context_pos in range(left, right):
            if context_pos == center_pos:
                continue

            context_word = tokens[context_pos]

            if context_word not in word_to_id:
                continue

            context_id = word_to_id[context_word]
            pairs.append((center_id, context_id))

    return pairs


def sample_negative(context_id, vocab_size):
    """
    Uniformly sample one negative id different from context_id.
    """
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1")

    negative_id = random.randrange(vocab_size)

    while negative_id == context_id:
        negative_id = random.randrange(vocab_size)

    return negative_id


def make_training_triples(pairs, vocab_size):
    """
    Convert positive center-context pairs into
    center-positive-negative triples.
    """
    triples = []

    for center_id, positive_id in pairs:
        negative_id = sample_negative(positive_id, vocab_size)
        triples.append((center_id, positive_id, negative_id))

    return triples