import inngest
import src.services.public as public_svcs
from src import schemas
from src.core import inngest_client, app


@inngest_client.create_function(
    fn_id="ingest-documents",
    trigger=inngest.TriggerEvent(event="rag/ingest-documents"),
    retries=0,
)
async def ingest_documents(ctx: inngest.Context) -> schemas.IngestionResponse:
    return public_svcs.ingest_documents(ctx).model_dump()
