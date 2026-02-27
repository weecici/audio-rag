import os
import asyncio
import re
from google import genai
from google.genai import types
from typing import Literal
from functools import lru_cache
from app.core import config
from app.utils import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

transcript_chunking_template = """
You are an expert at segmenting transcripts into contextually coherent chunks for RAG.

Parameters:
- `max_token`: {max_token}
- `lang`: {lang}

Rules:
1. Divide the transcript into logical chunks (each covering a single topic or tight set of ideas).
2. DO NOT summarize, paraphrase, or change the meaning. Keep the exact sentences, only fixing obvious ASR typos.
3. CRITICAL: Every sentence from the source transcript must be included in your output exactly once. Do not drop or skip any parts of the transcript.
4. Compute each chunk's `start_time` and `end_time` (in seconds) from the original timestamps. Round to the nearest integer.
5. Provide a short, highly descriptive `title` for each chunk. The title must be filesystem-safe (no slashes).
6. Format your output EXACTLY as follows for each chunk, separated by ten equals signs:

<title> | <start_time> | <end_time>
++++++++++
<chunk_text>

==========

(Repeat for all chunks. Do not use markdown blocks, do not add extra commentary.)

Now process the transcript below using these rules:
{transcript}
"""

document_chunking_template = """
You are an expert at segmenting text documents into contextually coherent chunks for RAG.

Parameters:
- `max_token`: {max_token}
- `lang`: {lang}

Rules:
1. Divide the document into logical chunks (each covering a single topic or tight set of ideas).
2. DO NOT summarize, paraphrase, or change the meaning. Keep the exact sentences.
3. CRITICAL: Every sentence from the source document must be included in your output exactly once. Do not drop or skip any parts of the document.
4. Provide a short, highly descriptive `title` for each chunk. The title must be filesystem-safe (no slashes).
5. Format your output EXACTLY as follows for each chunk, separated by ten equals signs:

<title>
++++++++++
<chunk_text>

==========

(Repeat for all chunks. Do not use markdown blocks, do not add extra commentary.)

Now process the document below using these rules:
{document}
"""


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    if not config.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    logger.info("Initializing Google Gemini client")
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return client


_semaphore = None
limit = 5


def _get_semaphore():
    """Lazy-init a semaphore to cap outbound concurrency and avoid rate limits."""
    global _semaphore, limit
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(limit)
    return _semaphore


def parse_response_into_chunks(
    response_text: str, text_type: Literal["transcript", "document"] = "transcript"
) -> list[tuple[str, str]]:

    chunk_separator = "\n=========="
    chunks = response_text.strip().split(chunk_separator)
    title_template = (
        "{title} || {start_time} || {end_time}"
        if text_type == "transcript"
        else "{title}"
    )
    parsed_chunks = []

    for chunk in chunks:
        title_line = "unknown"
        try:
            if "\n++++++++++\n" not in chunk:
                continue
            title_line, chunk_text = chunk.split("\n++++++++++\n", 1)
            title_line = title_line.replace("/", "-")  # sanitize filename
            if text_type == "transcript":
                title_parts = title_line.split(" | ")
                if len(title_parts) != 3:
                    logger.warning(f"Unexpected title format: {title_line}")
                    # fallback format if split fails
                    title = title_template.format(
                        title=title_line.strip(), start_time="0", end_time="0"
                    )
                else:
                    title_str, start_time, end_time = title_parts
                    title = title_template.format(
                        title=title_str.strip(),
                        start_time=start_time.strip(),
                        end_time=end_time.strip(),
                    )
            else:  # document
                title = title_template.format(title=title_line.strip())

            parsed_chunks.append((title, chunk_text.strip()))

        except Exception as e:
            logger.error(f"Error parsing chunk {title_line}: {e}")
            continue
    return parsed_chunks


def _fallback_chunk_transcript(raw_text: str, max_tokens: int) -> list[tuple[str, str]]:
    logger.info("Using rule-based fallback for transcript chunking.")
    lines = raw_text.split("\n")
    chunks = []
    current_chunk_lines = []
    current_length = 0
    start_t = "0"
    end_t = "0"
    chunk_start_t = None

    timestamp_re = re.compile(r"\[(\d+(?:\.\d+)?)s\s*-\s*(\d+(?:\.\d+)?)s\]")

    for line in lines:
        if not line.strip():
            continue

        m = timestamp_re.search(line)
        if m:
            if chunk_start_t is None:
                chunk_start_t = str(round(float(m.group(1))))
            end_t = str(round(float(m.group(2))))

        current_chunk_lines.append(line)
        # Using a rough approximation of tokens (chars / 4)
        current_length += len(line) // 4

        if current_length >= max_tokens:
            title = f"Transcript Segment || {chunk_start_t or start_t} || {end_t}"
            chunks.append((title, "\n".join(current_chunk_lines)))
            current_chunk_lines = []
            current_length = 0
            chunk_start_t = None

    if current_chunk_lines:
        title = f"Transcript Segment || {chunk_start_t or start_t} || {end_t}"
        chunks.append((title, "\n".join(current_chunk_lines)))

    return chunks


def _fallback_chunk_document(raw_text: str, max_tokens: int) -> list[tuple[str, str]]:
    logger.info("Using rule-based fallback for document chunking.")
    chunk_size = max_tokens * 4
    chunk_overlap = min(200, chunk_size // 4)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
    )
    docs = splitter.create_documents([raw_text])

    chunks = []
    for i, doc in enumerate(docs):
        title = f"Document Segment {i + 1}"
        chunks.append((title, doc.page_content))
    return chunks


async def chunk_text(
    raw_text: str,
    text_type: Literal["transcript", "document"] = "transcript",
    save_outputs: bool = True,
    output_dir: str = config.CHUNKED_TRANSCRIPT_STORAGE_PATH,
    max_tokens: int = config.MAX_TOKENS,
) -> list[tuple[str, str]]:
    """Return a list of tuples: (title, chunk_text)"""

    try:
        client = _get_client()

        if text_type == "transcript":
            prompt = transcript_chunking_template.format(
                transcript=raw_text, max_token=max_tokens, lang="vi"
            )
        elif text_type == "document":
            prompt = document_chunking_template.format(
                document=raw_text, max_token=max_tokens, lang="vi"
            )
        else:
            raise ValueError(f"Invalid text_type: {text_type}")

        logger.info(f"Sending LLM chunking request for type **{text_type}**...")

        model_config = types.GenerateContentConfig(
            system_instruction="You are an expert in chunking texts into smaller, coherent chunks for information retrieval systems. And you must follow the rules in the prompt strictly.",
        )
        sem = _get_semaphore()
        async with sem:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=config.CHUNKING_MODEL,
                contents=prompt,
                config=model_config,
            )

        if response is None or response.text is None:
            raise RuntimeError("No response from Gemini model")

        chunks = parse_response_into_chunks(
            response_text=response.text, text_type=text_type
        )

        # Fallback if parsing failed or gave zero chunks
        if not chunks:
            raise RuntimeError("LLM returned zero chunks or parsing failed completely.")

    except Exception as e:
        logger.error(f"Error in LLM chunking: {e}. Falling back to rule-based chunker.")
        if text_type == "transcript":
            chunks = _fallback_chunk_transcript(raw_text, max_tokens)
        else:
            chunks = _fallback_chunk_document(raw_text, max_tokens)

    if save_outputs and chunks:
        os.makedirs(output_dir, exist_ok=True)
        for title, chunk_text in chunks:
            safe_title = title.replace("/", "-").replace(":", "-").replace("|", "-")
            chunk_path = os.path.join(output_dir, f"{safe_title}.txt")
            with open(chunk_path, "w", encoding="utf-8") as cf:
                cf.write(chunk_text)

    return chunks
