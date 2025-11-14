from functools import lru_cache
from cerebras.cloud.sdk import Cerebras
from src.core import config
from src.utils import logger


@lru_cache(maxsize=1)
def _get_llm_client() -> Cerebras:
    if not config.CEREBRAS_API_KEY:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    logger.info("Initializing Cerebras LLM client")
    client = Cerebras(api_key=config.CEREBRAS_API_KEY)
    return client


def generate(prompts: list[str], model: str) -> list[str]:
    client = _get_llm_client()
    messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]

    responses = []
    for messages in messages_list:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        responses.append(response.choices[0].message.content)
    return responses
