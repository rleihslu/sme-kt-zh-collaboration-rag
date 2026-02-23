from textwrap import dedent
from typing import Sequence

from conversational_toolkit.llms.base import LLM, LLMMessage, Roles
from conversational_toolkit.vectorstores.base import ChunkRecord


async def make_query_standalone(llm: LLM, history: list[LLMMessage], query: str) -> str:
    template_query_standalone = dedent("""
        Objective: Your task is to analyze the input query and the provided conversation history. You need to reformulate the query so that it becomes independent of the conversation history, allowing anyone unfamiliar with the history to understand the query without needing additional context. However, if the input query is completely unrelated to the conversation history or already stands on its own without needing context from the history, you should not rephrase it.

        Input:

            User Query: {query}
            Conversation History:
                {chat_history}

        Instructions:

            Step 1: Examine the current user query in the context of the provided conversation history.
            Step 2: Determine if the query is dependent on the conversation history for its clarity and relevance. If it is, proceed to reformulate the query.
                When reformulating, ensure that the new query is clear, concise, and can be understood without prior knowledge of the conversation history.
                Use general terms instead of pronouns or references that only make sense with the history.
            Step 3: If the query is completely independent of the conversation history or does not require context from the history for a new user to understand, do not reformulate it.
            Step 4: Provide the reformulated or original query as the output.

        Example:

            User Query: "But what about its effects on global trade?"

            Conversation History:
                User: "Can you tell me about the recent changes in oil prices?"
                Assistant: "Recent changes in oil prices have been significant due to various geopolitical events and shifts in supply and demand dynamics."

            "What are the effects of recent changes in oil prices on global trade?"

        Output:

        If the query was reformulated based on its dependence on the conversation history, your response should look like this:

            [Insert the reformulated query here, making it independent of the conversation history.]

        If the query did not require reformulation and is already clear and independent:

            [Insert the original query here, indicating it was already clear and independent of the conversation history.]
    """)
    chat_history = "\n".join([f"{message.role}: {message.content}" for message in history])
    conversation = [
        LLMMessage(
            role=Roles.SYSTEM,
            content="You are a helpful assistant that transforms a message from a user to be independent from the conversation history given.",
        ),
        LLMMessage(
            role=Roles.USER,
            content=template_query_standalone.format(query=query, chat_history=chat_history),
        ),
    ]
    reformulated_query = (await llm.generate(conversation)).content
    return reformulated_query


async def query_expansion(query: str, llm: LLM, expansion_number: int = 2) -> list[str]:
    template_query_expansion = """
        Generate multiple search queries related to: {query}, and translate them in english if they are not already in english. Only output {expansion_number} queries in english.
        OUTPUT ({expansion_number} queries):
    """
    conversation = [
        LLMMessage(
            role=Roles.SYSTEM,
            content="You are a focused assistant designed to generate multiple, relevant search queries based solely on a single input query. Your task is to produce a list of these queries in English, without adding any further explanations or information.",
        ),
        LLMMessage(
            role=Roles.USER, content=template_query_expansion.format(query=query, expansion_number=expansion_number)
        ),
    ]

    generated_queries = (await llm.generate(conversation)).content.strip().split("\n")
    return generated_queries


async def hyde_expansion(query: str, llm: LLM) -> str:
    conversation = [
        LLMMessage(
            role=Roles.SYSTEM,
            content="You are a helpful assistant. Provide an example of answer to the provided query. Only output an hypothetical explanation to the query.",
        ),
        LLMMessage(role=Roles.USER, content=query),
    ]
    hyde_expansion_message = (await llm.generate(conversation)).content
    return hyde_expansion_message


def reciprocal_rank_fusion(search_results: Sequence[Sequence[ChunkRecord]], k: int = 60) -> list[ChunkRecord]:
    """
    Applies Reciprocal Rank Fusion (RRF) to a list of search results from different sources.

    Parameters:
    - search_results: A list of tuples where each tuple contains a source name and a ranked list of ChunkRecord objects.
    - k: A constant that dampens the rank influence, default is 60.

    Returns:
    - A list of tuples, each containing a source name and a Chunk object, sorted by the fused score.
    """
    fused_scores = {}
    chunk_map = {}

    for chunks in search_results:
        for rank, chunk in enumerate(chunks, start=1):
            score = float(1 / (k + rank))
            if chunk.id not in fused_scores:
                fused_scores[chunk.id] = score
            else:
                fused_scores[chunk.id] += score
            chunk_map[chunk.id] = chunk

    sorted_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

    return [chunk_map[chunk_id] for chunk_id, _ in sorted_results]


def build_query_with_chunks(user_query: str = "", chunks: list[ChunkRecord] | None = None) -> str:
    """
    Constructs a query string for an LLM, embedding relevant chunk content inside XML tags.

    Parameters:
    - user_query: The original user query string (optional).
    - chunks: List of Chunk objects containing source information and content.

    Returns:
    - A formatted string with the user query (if provided) and relevant sources in XML format.
    """
    if not chunks:
        sources_xml = "No sources found."
    else:
        sources_xml = "\n".join(f'<source id="{chunk.id}">\n{chunk.content}\n</source>' for chunk in chunks)

    return dedent(
        f"""
        User Query: {user_query}\n
        Here are the sources I found for you:
        {sources_xml}
        """
    )
