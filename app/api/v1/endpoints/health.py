from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness probe")
async def ready() -> dict[str, str]:
    # TODO: check downstream dependencies (Milvus, model availability, etc.)
    return {"status": "ready"}
