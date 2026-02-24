from underthesea import word_tokenize
import string
from pathlib import Path

_current_dir = Path(__file__).parent
_stopwords_path = _current_dir / "vietnamese-stopwords.txt"

with open(_stopwords_path, "r", encoding="utf-8") as f:
    _stopwords: set[str] = set(line.strip() for line in f)

_punctuations: set[str] = set([p for p in string.punctuation])


def tokenize(
    texts: list[str],
) -> list[list[str]]:
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]

    # Remove punctuation
    punc_removed_tok_lists = [
        [token for token in tokens if token not in _punctuations]
        for tokens in tokenized_texts
    ]

    # Remove stopwords
    sw_removed_tok_lists = [
        [token for token in tokens if token not in _stopwords]
        for tokens in punc_removed_tok_lists
    ]

    return sw_removed_tok_lists
