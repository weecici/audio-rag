import uuid
import anyio
from fastapi import UploadFile
from pathlib import Path


async def save_upload(upload: UploadFile, base_dir: Path) -> tuple[str, int]:
    original_name = (upload.filename or "").strip()
    safe_name = Path(original_name).name if original_name else "upload"
    dest = base_dir / f"{uuid.uuid5(uuid.NAMESPACE_DNS, safe_name).hex}"

    def _copy() -> int:
        size = 0
        upload.file.seek(0)
        with dest.open("wb") as out:
            while True:
                chunk = upload.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                out.write(chunk)
        return size

    size_bytes = await anyio.to_thread.run_sync(_copy)
    await upload.close()
    return str(dest), size_bytes
