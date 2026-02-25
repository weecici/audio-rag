from functools import lru_cache
from cerebras.cloud.sdk import Cerebras
from app.core import config
from app.util import logger


@lru_cache(maxsize=1)
def _get_client() -> Cerebras:
    if not config.CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    logger.info("Initializing Cerebras LLM client")
    client = Cerebras(api_key=config.CEREBRAS_API_KEY)
    return client


def generate(prompts: list[str], model: str) -> list[str]:
    client: Cerebras = _get_client()
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

    responses = []
    for messages in messages_list:
        response = client.chat.completions.create(
            model=model, messages=messages, reasoning_effort="high"
        )
        responses.append(response.choices[0].message.content)
    return responses
