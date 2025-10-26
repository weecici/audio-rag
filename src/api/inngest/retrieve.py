import inngest
import src.services.public as public_svcs
from src import schemas
from src.core import inngest_client, app


@inngest_client.create_function(
    fn_id="retrieve-documents",
    trigger=inngest.TriggerEvent(event="rag/retrieve-documents"),
    retries=0,
)
async def retrieve_documents(ctx: inngest.Context) -> schemas.RetrievalResponse:
    return public_svcs.retrieve_documents(ctx).model_dump()
