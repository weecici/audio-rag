from app import schema

prompt_template = """
You are an AI assistant that helps users by providing detailed answers based on the context provided. Use the following context to answer the question below. If the context does not contain the answer, respond with "I don't know".

CONTEXT DOCUMENTS:
{context}

QUESTION:
{query}
"""


def get_augmented_prompts(
    queries: list[str], contexts: list[list[schema.RetrievedDocument]]
) -> list[str]:
    if len(queries) != len(contexts):
        raise ValueError(
            "The number of queries must match the number of context lists."
        )

    formatted_contexts = [
        "\n".join(
            f"- {doc.payload.text} (Source: {doc.payload.metadata.title})"
            for doc in context
        )
        for context in contexts
    ]

    augmented_prompts = [
        prompt_template.format(context=context, query=query)
        for query, context in zip(queries, formatted_contexts)
    ]

    return augmented_prompts
