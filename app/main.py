from app.core import app
from app.api.v1 import api_v1

app.include_router(router=api_v1, prefix="/api/v1")
