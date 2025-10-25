import inngest.fast_api
from src.core import inngest_client, app
from src.api.inngest import inngest_functions

inngest.fast_api.serve(app=app, client=inngest_client, functions=inngest_functions)
