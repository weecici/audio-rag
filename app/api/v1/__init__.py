from fastapi import APIRouter

from .endpoints import conversations, files, health, jobs, search, openai_compat

router = APIRouter(prefix="/v1")
router.include_router(health.router)
router.include_router(files.router)
router.include_router(jobs.router)
router.include_router(search.router)
router.include_router(conversations.router)
router.include_router(openai_compat.router)
