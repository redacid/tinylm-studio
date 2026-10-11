"""Grounds a chat request in retrieved chunks so the model answers from them and cites sources."""

from dataclasses import replace

from engine.chat import ChatRequest, Message
from engine.rag.store import Hit, KnowledgeStore

INSTRUCTIONS = (
    "Answer the user's question using the numbered sources below. "
    "After each fact, cite its source number in square brackets, like [1]. "
    "If the sources do not contain the answer, say so and answer from general knowledge, "
    "clearly marked as such. Answer in the language of the user's question."
)


def last_user_text(request: ChatRequest) -> str:
    return next((m.content for m in reversed(request.messages) if m.role == "user"), "")


def ground(request: ChatRequest, hits: list[Hit]) -> ChatRequest:
    sources = "\n\n".join(f"[{i}] {hit.document}\n{hit.text}" for i, hit in enumerate(hits, 1))
    system = f"{INSTRUCTIONS}\n\nSources:\n\n{sources}"
    messages = list(request.messages)
    if messages and messages[0].role == "system":
        system = f"{messages[0].content}\n\n{system}"
        messages = messages[1:]
    return replace(request, messages=[Message("system", system), *messages])


def retrieve(
    knowledge: KnowledgeStore, collection: str, request: ChatRequest, k: int = 4
) -> tuple[ChatRequest, list[dict]]:
    """Returns the grounded request and the sources shown to the client, numbered from 1."""
    hits = knowledge.search(collection, last_user_text(request), k)
    if not hits:
        return request, []
    sources = [{"index": i, **hit.to_dict()} for i, hit in enumerate(hits, 1)]
    return ground(request, hits), sources
