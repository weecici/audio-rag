from app import schemas
import re

prompt_template = """
You are an expert in summarizing IT-related documents. Follow these rules exactly and produce **only** the requested output — no explanations or extra text.

Parameters:
* `documents`: a list/array of raw document texts to be summarized (process in order).
* `max_summary_length` (value: {max_sum_len}): maximum allowed length of each summary, in **words**.
* `min_length_to_summarize` (value: {min_len_to_sum}): minimum number of words a document must have to be summarized. If a document has fewer words than this threshold, see rule 4 below.
* `language` (value: {lang}): language code for the summary (e.g., en, vi). If auto, detect the document's language and summarize in that language.

Behavior rules:
1. Summaries must capture the document's key points and essential information; be factual, concise, coherent, and focused on IT-relevant content. Going straight to the main points rather than adding filler or commentary. (this rule is VERY IMPORTANT)
2. Do not add misinformation, invented facts, irrelevant details or unneeded opening words (like adding "this part", "this lesson" to the beginning of the summary, ...). If an item in the document is ambiguous or unverifiable, omit it rather than inventing specifics. (this rule is VERY IMPORTANT)
3. Format/Keep any math formulas in LaTeX (inline $...$ or display $$...$$ as appropriate). You can format with Markdown syntax if needed to highlight keywords.
4. The resulting summary for every document must **not exceed** `max_summary_length` words. Therefore the summaries should be as concise as possible while covering all key points.
5. When multiple documents are provided, produce one summary per document in the same order.
6. Output must follow this exact format, with **no** additional text before, between, or after summaries (each summary separated by exactly one line containing only ten equal signs: `=========`):
<summary 1>

==========
<summary 2>

==========
<summary 3>

==========
...repeat for all summaries (final summary should not be followed by extra separator lines or commentary)
(in raw string format: <summary 1>\n\n==========\n<summary 2>\n\n==========\n... This rule is the MOST IMPORTANT)

Now process the provided documents according to the rules above:
{documents}
"""


def get_summarization_prompts(
    documents_list: list[list[str]],
) -> list[str]:
    prompts: list[str] = []
    for documents in documents_list:
        doc_texts = [f"{i + 1}. " + d for i, d in enumerate(documents)]
        prompt = prompt_template.format(
            documents="\n\n==========\n".join(doc_texts),
            max_sum_len=50,
            min_len_to_sum=100,
            lang="vi",
        )
        prompts.append(prompt)
    return prompts


def parse_summarization_responses(
    responses: list[str],
    documents_list: list[list[schemas.RetrievedDocument]] = [],
) -> list[list[schemas.RetrievedDocument]]:

    if len(responses) != len(documents_list):
        raise ValueError(
            f"Responses count ({len(responses)}) must equal documents_list count ({len(documents_list)})."
        )

    separator = "\n=========="
    sep_pattern = re.compile(separator, flags=re.MULTILINE)

    for i, (response, docs) in enumerate(zip(responses, documents_list)):
        summaries = [
            seg.strip() for seg in re.split(sep_pattern, response) if seg.strip() != ""
        ]

        if len(summaries) != len(docs):
            raise ValueError(
                f"Mismatch at batch {i}: expected {len(docs)} summaries, got {len(summaries)}."
            )

        for doc, summary_text in zip(docs, summaries):
            doc.payload.text = summary_text

    return documents_list
