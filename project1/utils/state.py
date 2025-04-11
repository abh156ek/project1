# src/utils/state.py

from typing_extensions import Annotated, Sequence, TypedDict
import operator
from langchain_core.messages import BaseMessage


def merge_dicts(a: dict[str, any], b: dict[str, any]) -> dict[str, any]:
    """Merge two dictionaries, with values from b overwriting those from a."""
    return {**a, **b}


class AgentState(TypedDict):
    """Typed state object used by LangGraph agent nodes."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, any], merge_dicts]
    metadata: Annotated[dict[str, any], merge_dicts]
