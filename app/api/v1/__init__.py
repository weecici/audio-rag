from fastapi import APIRouter
from .ingest import router as ingest_router
from .retrieve import router as retrieve_router
from .generate import router as generate_router

api_v1 = APIRouter()
api_v1.include_router(ingest_router)
api_v1.include_router(retrieve_router)
api_v1.include_router(generate_router)
