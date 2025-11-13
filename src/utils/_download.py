from __future__ import annotations

import os
import yt_dlp as ytdlp
from pathlib import Path
from typing import Any, Optional, Sequence, Union
from ._logging import logger
from src.core import config


def _ensure_list(urls: Union[str, Sequence[str]]) -> list[str]:
    if isinstance(urls, str):
        return [urls]

    return [u for u in urls if isinstance(u, str) and u.strip()]


def download_audio(
    urls: Union[str, Sequence[str]],
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
) -> list[Path]:

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

    downloaded_files: list[Path] = []

    def _hook(d: dict[str, Any]) -> None:
        if d.get("status") == "finished":
            fn = d.get("filename")
            if fn:
                downloaded_files.append(Path(fn))
                logger.info(f"Downloaded: {fn}")

    ydl_opts.setdefault("progress_hooks", []).append(_hook)

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
        seen = set()
        unique_files = []
        for p in downloaded_files:
            if p not in seen:
                unique_files.append(p)
                seen.add(p)

    return unique_files
