from collections import Counter
from app import schema
from app.util import tokenize


def build_inverted_index(
    texts: list[str],
    doc_ids: list[str],
) -> tuple[dict[str, schema.TermEntry], dict[str, int]]:
    """Returns postings list and document lengths for the given texts."""

    postings_list: dict[str, schema.TermEntry] = {}
    doc_lens: dict[str, int] = {}

    tokenized_docs = tokenize(texts=texts)

    # creating postings list
    for doc_id, tokens in zip(doc_ids, tokenized_docs):
        term_counts = Counter(tokens)
        for token, term_freq in term_counts.items():
            doc_lens[doc_id] = doc_lens.get(doc_id, 0) + term_freq
            if token not in postings_list:
                postings_list[token] = schema.TermEntry(doc_freq=0, postings=[])

            postings_list[token].postings.append(
                schema.PostingEntry(doc_id=doc_id, term_freq=term_freq)
            )

    # compute document frequencies
    for term, term_entry in postings_list.items():
        postings_list[term].doc_freq = len(term_entry.postings)

    return postings_list, doc_lens
