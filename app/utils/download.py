from __future__ import annotations

import os
import yt_dlp as ytdlp
from pathlib import Path
from typing import Any, Optional, Sequence, Union
from .logging import logger
from app.core import config


def _ensure_list(urls: Union[str, list[str]]) -> list[str]:
    if isinstance(urls, str):
        return [urls]
    return urls


def download_audio(
    urls: Union[str, list[str]],
    out_dir: Union[str, Path] = config.AUDIO_STORAGE_PATH,
    filename_template: str = "%(title)s $ %(id)s.%(ext)s",
    codec: str = "wav",
    sample_rate: Optional[int] = 16000,
    quality_kbps: str = "192",
    overwrite: bool = True,
    allow_playlist: bool = False,
    rate_limit_bytes: Optional[int] = None,
    retries: int = 3,
    timeout: Optional[int] = 30,
    extra_opts: Optional[dict[str, Any]] = None,
) -> list[str]:

    urls_list = _ensure_list(urls)
    if not urls_list:
        logger.warning("download_audio called with empty URL list.")
        return []

    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = os.path.join(str(base_dir), filename_template)

    # Prepare post-processor args for ffmpeg
    pp_args: list[str] = ["-y"]
    if sample_rate:
        pp_args.extend(["-ar", str(sample_rate)])

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": not allow_playlist,
        "ignoreerrors": "only_download",
        "retries": retries,
        "socket_timeout": timeout,
        "continuedl": True,
        "overwrites": overwrite,
        # Extract and convert to desired codec using ffmpeg
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": codec,
                "preferredquality": quality_kbps,
            }
        ],
        # Extra args for ffmpeg (applied to postprocessing), include sample rate
        "postprocessor_args": pp_args,
        "logger": logger,
    }

    if rate_limit_bytes is not None and rate_limit_bytes > 0:
        ydl_opts["ratelimit"] = rate_limit_bytes

    if extra_opts:
        ydl_opts.update(extra_opts)

    downloaded_files: list[str] = []

    def _prog_hook(d: dict[str, Any]) -> None:
        if d.get("status") == "finished":
            logger.debug("Download finished; waiting for post-processing...")

    def _pp_hook(d: dict[str, Any]) -> None:
        if d.get("status") != "finished" or d.get("postprocessor") != "MoveFiles":
            return

        info = d.get("info_dict") or {}
        final_fp = info.get("filepath")

        if not final_fp:
            # Fall back to pre-PP filename with desired codec extension
            src = info.get("_filename") or info.get("filename")
            if src:
                final_fp = str(Path(src).with_suffix(f".{codec.lower()}"))
        if final_fp:
            downloaded_files.append(final_fp)
            logger.info(f"Post-processed: {final_fp}")

    ydl_opts.setdefault("progress_hooks", []).append(_prog_hook)
    ydl_opts.setdefault("postprocessor_hooks", []).append(_pp_hook)

    logger.debug(
        f"Starting audio download for {len(urls_list)} url(s) -> {base_dir} with codec={codec}, quality={quality_kbps}kbps"
    )

    try:
        with ytdlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download(urls_list)
    except Exception as e:
        logger.exception(f"yt-dlp download failed: {e}")
    finally:
        # Deduplicate keeping order
        seen: set[str] = set()
        unique_files: list[str] = []
        for p in downloaded_files:
            if p not in seen:
                unique_files.append(p)
                seen.add(p)

    return unique_files
