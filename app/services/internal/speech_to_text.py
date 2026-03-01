"""Speech-to-text transcription using faster-whisper BatchedInferencePipeline."""

import threading
import torch
from functools import lru_cache
from pathlib import Path
from faster_whisper import WhisperModel, BatchedInferencePipeline

from app.core.config import settings
from app.core.logging import logger

# Serialise GPU access – only one transcription batch at a time.
_gpu_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_batched_model() -> BatchedInferencePipeline:
    """Return a cached ``BatchedInferencePipeline`` singleton.

    The model is created on the best available device (CUDA > CPU).
    CTranslate2's ``load_model`` / ``unload_model`` are used later to
    manage VRAM between transcription batches.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"

    logger.info(
        f"Loading speech-to-text model: "
        f"faster-whisper-{settings.SPEECH_TO_TEXT_MODEL_SIZE} "
        f"(device={device}, compute_type={compute_type})"
    )
    model = WhisperModel(
        settings.SPEECH_TO_TEXT_MODEL_SIZE,
        device=device,
        compute_type=compute_type,
    )
    return BatchedInferencePipeline(model=model)


def _transcribe_single(
    batched_model: BatchedInferencePipeline,
    audio_path: Path,
    *,
    language: str | None = None,
    batch_size: int = 4,
) -> str:
    """Transcribe one audio file and return timestamped transcript text."""
    segments, info = batched_model.transcribe(
        str(audio_path),
        language=language,
        batch_size=batch_size,
    )

    # lines: list[str] = []
    # for seg in segments:
    #     lines.append(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text.strip()}")
    transcript = " ".join(seg.text.strip() for seg in segments)

    logger.info(
        f"Transcribed {audio_path.name}: "
        f"lang={info.language} ({info.language_probability:.0%}), "
        f"{len(transcript)} chars"
    )
    return transcript


def parse_audio_to_text(
    audio_paths: list[Path],
    out_dir: Path | None = None,
    *,
    language: str | None = None,
    batch_size: int = 4,
) -> list[Path]:
    """Transcribe audio files and return paths to transcript ``.txt`` files.

    This function is designed to be called from ``asyncio.run_in_executor``
    so the event loop is not blocked.

    Lifecycle per call:
      1. Acquire ``_gpu_lock``.
      2. Ensure CTranslate2 weights are on device (``load_model``).
      3. Transcribe each file via ``BatchedInferencePipeline``.
      4. Unload weights from CUDA and release VRAM.

    Args:
        audio_paths: Paths to audio files (.mp3, .wav, .ogg, .flac, .aac).
        out_dir: Directory for transcript files.  Defaults to
            ``settings.TRANSCRIPT_STORAGE_PATH``.
        language: ISO-639-1 language code hint (e.g. ``"vi"``).
            ``None`` lets Whisper auto-detect.
        batch_size: Number of audio segments decoded in parallel by
            the batched pipeline.

    Returns:
        List of transcript ``.txt`` file paths (same order as input,
        minus failures).
    """
    if not audio_paths:
        return []

    if out_dir is None:
        out_dir = Path(settings.TRANSCRIPT_STORAGE_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    batched_model = _get_batched_model()
    ct2_model = batched_model.model.model  # ctranslate2.models.Whisper

    transcript_paths: list[Path] = []

    with _gpu_lock:
        # Ensure weights are on-device (no-op if already loaded).
        ct2_model.load_model()

        try:
            for audio_path in audio_paths:
                try:
                    logger.info(f"Transcribing: {audio_path.name}")
                    transcript = _transcribe_single(
                        batched_model,
                        audio_path,
                        language=language,
                        batch_size=batch_size,
                    )

                    transcript_path = out_dir / f"{audio_path.stem}.txt"
                    transcript_path.write_text(transcript, encoding="utf-8")
                    transcript_paths.append(transcript_path)

                    logger.info(f"Transcript saved: {transcript_path}")
                except Exception as exc:
                    logger.error(f"Failed to transcribe {audio_path.name}: {exc}")
                    continue
        finally:
            # Free VRAM after the batch is done.
            if ct2_model.device == "cuda":
                ct2_model.unload_model()
                torch.cuda.empty_cache()
                logger.debug("Whisper model unloaded from CUDA, VRAM freed.")

    return transcript_paths
