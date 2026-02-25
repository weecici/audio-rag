import torch
import os
from pathlib import Path
from functools import lru_cache
from typing import Union
from app.core import config
from app.util import logger
from faster_whisper import WhisperModel, BatchedInferencePipeline


@lru_cache(maxsize=1)
def _get_s2t_batched_model() -> BatchedInferencePipeline:
    logger.info(
        f"Loading speech to text model: faster-whisper-{config.SPEECH2TEXT_MODEL}"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(
        config.SPEECH2TEXT_MODEL, device=device, compute_type="float16"
    )
    batched_model = BatchedInferencePipeline(model=model)
    return batched_model


def _ensure_list(paths: Union[str, list[str]]) -> list[str]:
    if isinstance(paths, str):
        return [paths]
    return paths


def transcribe_audio(
    audio_paths: Union[str, list[str]],
    out_dir: str = config.TRANSCRIPT_STORAGE_PATH,
    language: str = "vi",
    batch_size: int = 4,
) -> list[str]:

    audio_paths = _ensure_list(audio_paths)
    if not audio_paths:
        logger.warning("No valid audio files provided to transcribe_audio")
        return []

    batched_model = _get_s2t_batched_model()
    batched_model.model.model.load_model()

    os.makedirs(out_dir, exist_ok=True)

    transcripts = []
    filepaths = []
    for audio_path in audio_paths:
        try:
            logger.info(f"Transcribing audio file: {audio_path}")

            segments, info = batched_model.transcribe(
                audio_path, language=language, batch_size=batch_size
            )

            transcript = ""
            for segment in segments:
                s, e, t = segment.start, segment.end, segment.text.strip()
                transcript += f"[{s:.2f}s - {e:.2f}s] {t}\n"

            # Write out the transcript
            audio_filename = Path(audio_path).stem
            transcript_path = os.path.join(out_dir, f"{audio_filename}.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)

            filepaths.append(transcript_path)

            logger.info(f"Completed saving transcription for: {audio_path}")
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            transcripts.append("")
            continue

    if batched_model.model.model.device == "cuda":
        batched_model.model.model.unload_model()
        torch.cuda.empty_cache()

    return filepaths
