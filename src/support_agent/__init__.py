"""Customer Support Agent public package API."""

__version__ = "0.1.0"
__all__ = ["graph", "SupportState", "tools"]


def __getattr__(name: str):
    """Load LangChain/LangGraph dependencies only when an API component is requested."""
    if name == "graph":
        from .agent import graph

        return graph
    if name == "SupportState":
        from .state import SupportState

        return SupportState
    if name == "tools":
        from .tools import tools

        return tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
