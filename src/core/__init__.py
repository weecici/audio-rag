import inngest
import logging
from inngest import Inngest
from fastapi import FastAPI

inngest_client = Inngest(
    app_id="cs419-rag",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)
app = FastAPI()
