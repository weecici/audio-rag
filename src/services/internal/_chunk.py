import os
import re
from google import genai
from typing import Union, Any
from functools import lru_cache
from src.core import config
from src.utils import logger

MODEL_ID = "gemini-flash-latest"


prompt_template = """
You are an expert at chunking transcripts of audio files for information retrieval systems. Follow these rules exactly and produce only the requested output — no explanations or extra text.

Parameters (replace placeholders):
- max_token (default: {max_token}): maximum tokens per chunk (model tokens, e.g., using tiktoken cl100k_base). Approx: 1 token ≈ 0.75 words.
- raw_transcript (will be provided later): the raw transcript text.
- lang (default: {lang}): main language of the transcript. If omitted, detect language automatically.

Behavior rules:
1. Correct only non-substantive syntax errors: punctuation, obvious typos, unmatched quotes, and spacing. Do NOT paraphrase, summarize, condense, or change facts or technical expressions.
2. Remove all original timestamps and timestamp markers from the chunk text. Keep speaker labels only if they are essential; otherwise remove them.
3. Compute each chunk's start_time and end_time (in seconds) from the original timestamps and display them in the chunk title. Round to the nearest integer second (0.5 → round up). 
4. Chunk boundaries must be at natural linguistic boundaries (sentence boundaries). Do not break sentences across chunks, except when a single sentence exceeds {max_token} tokens (see rule 8).
5. Each chunk must be coherent and contextually relevant. Aim for semantic unity (a chunk should cover a single topic/idea or tightly related set of sentences).
6. Maximum chunk size is {max_token} tokens. Use the specified tokenizer to count tokens.
7. Title each chunk with a short topic name (in the transcript's main language, also mustn't contain some characters that are not allowed for filenames) followed by start and end times as integers, using the exact format:
   <title> | <start_time> | <end_time>
8. Fallback for very long sentences: if one sentence alone exceeds {max_token} tokens, split it at natural clause boundaries (commas, semicolons, conjunctions). Mark the split by appending " (continued)" to the first part and prefixing the second part with "(continued) ". Try to minimize meaning changes.
9. Output formatting: produce consecutive chunks separated by a line of ten equals signs "==========" and each chunk must follow exactly this template:

<title N> | <start_time N> | <end_time N>
++++++++++
<chunk_text N>

==========
(repeat for all chunks)

10. Do not use any markdown styling (no bold, italic, underline). Do not convert math to LaTeX. Keep math expressions verbatim.
11. Do not add any commentary, metadata, or notes outside the specified format. The assistant's response must contain only the chunks formatted as above.

Now process the transcript below using these rules **(slowly and carefully)**:
{transcript}

"""


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    if not config.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    logger.info("Initializing Google Gemini client")
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return client


def _ensure_list(paths: Union[str, list[str]]) -> list[str]:
    if isinstance(paths, str):
        return [paths]
    return paths


def parse_response_into_chunks(response_text: str) -> list[tuple[str, str]]:
    chunk_separator = "\n==========\n"
    chunks = response_text.strip().split(chunk_separator)
    parsed_chunks = []

    filename_template = "{title} $ {start_time} $ {end_time}.txt"

    for chunk in chunks:
        try:
            title_line, chunk_text = chunk.split("\n++++++++++\n", 1)
            title_parts = title_line.split(" | ")
            if len(title_parts) != 3:
                logger.warning(f"Unexpected title format: {title_line}")
                continue
            title, start_time, end_time = title_parts
            filename = filename_template.format(
                title=title.strip(),
                start_time=start_time.strip(),
                end_time=end_time.strip(),
            )
            parsed_chunks.append((filename, chunk_text.strip()))
        except Exception as e:
            logger.error(f"Error parsing chunk: {e}")
            continue
    return parsed_chunks


def chunk_transcripts(
    filepaths: Union[str, list[str]],
    output_dir: str = config.CHUNKED_TRANSCRIPT_STORAGE_PATH,
    max_tokens: int = config.MAX_TOKENS,
) -> list[str]:
    filepaths = _ensure_list(filepaths)
    client = _get_client()

    output_filepaths = []

    os.makedirs(output_dir, exist_ok=True)
    for filepath in filepaths:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                transcript = f.read()
            prompt = prompt_template.format(
                transcript=transcript, max_token=max_tokens, lang="vi"
            )

            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            print(response.text)

            chunks = parse_response_into_chunks(response.text)

            for filename, chunk_text in chunks:
                chunk_path = os.path.join(output_dir, filename)
                with open(chunk_path, "w", encoding="utf-8") as cf:
                    cf.write(chunk_text)

                output_filepaths.append(chunk_path)

        except Exception as e:
            logger.error(f"Error chunking transcript {filepath}: {e}")
            continue

    return output_filepaths
