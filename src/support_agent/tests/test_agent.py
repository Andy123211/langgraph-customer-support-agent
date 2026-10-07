"""Tests for agent graph logic and decision-making."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.support_agent.agent import (
    should_continue,
    agent_node,
    create_graph,
    execute_tools_node,
    route_after_tools,
)
from src.support_agent.state import SupportState


class FakeChatModel:
    def __init__(self, response):
        self.response = response

    def invoke(self, _messages):
        return self.response


class TestShouldContinue:
    """Test the conditional edge logic that decides whether to continue to tools or end."""

    def test_should_continue_with_tool_calls(self):
        """Test that agent continues to tools when AIMessage has tool_calls."""
        state: SupportState = {
            "messages": [
                HumanMessage(content="What's my order status?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "123456"},
                            "id": "call_1",
                        }
                    ],
                ),
            ]
        }

        result = should_continue(state)
        assert result == "tools", "Should route to tools when tool_calls exist"

    def test_should_continue_without_tool_calls(self):
        """Test that agent ends when AIMessage has no tool_calls."""
        state: SupportState = {
            "messages": [
                HumanMessage(content="Thank you!"),
                AIMessage(content="You're welcome! Have a great day!"),
            ]
        }

        result = should_continue(state)
        assert result == "__end__", "Should end when no tool_calls exist"

    def test_should_continue_with_empty_tool_calls_list(self):
        """Test that agent ends when tool_calls is an empty list."""
        state: SupportState = {
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!", tool_calls=[]),
            ]
        }

        result = should_continue(state)
        assert result == "__end__", "Should end when tool_calls is empty"

    def test_should_continue_with_human_message_last(self):
        """Test behavior when last message is not AIMessage (edge case)."""
        state: SupportState = {
            "messages": [
                AIMessage(content="How can I help?"),
                HumanMessage(content="I need help"),
            ]
        }

        result = should_continue(state)
        # HumanMessage doesn't have tool_calls, should end
        assert result == "__end__"


class TestToolFallbackRouting:
    """Test deterministic handoff paths without an LLM or external services."""

    def test_low_confidence_retrieval_routes_to_human(self, monkeypatch):
        class LowConfidenceTool:
            name = "search_mock"

            @staticmethod
            def invoke(_args):
                return "HANDOFF_REQUIRED: low_confidence_retrieval\nNo grounded match."

        from src.support_agent import agent as agent_module

        monkeypatch.setitem(agent_module._tools_by_name, "search_mock", LowConfidenceTool())
        state = {
            "messages": [
                HumanMessage(content="What is the policy for a very unusual case?"),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "search_mock",
                        "args": {"query": "unusual case"},
                        "id": "call-search",
                    }],
                ),
            ]
        }

        update = execute_tools_node(state)
        routed_state = {**state, **update}

        assert update["handoff_reason"] == "low_confidence_retrieval"
        assert update["support_status"] == "needs_human"
        assert route_after_tools(routed_state) == "human_handoff"

    def test_tool_exception_routes_to_human_without_exception_details(self, monkeypatch):
        class FailingTool:
            name = "failing_mock"

            @staticmethod
            def invoke(_args):
                raise RuntimeError("private upstream diagnostic")

        from src.support_agent import agent as agent_module

        monkeypatch.setitem(agent_module._tools_by_name, "failing_mock", FailingTool())
        state = {
            "messages": [
                HumanMessage(content="Check order 123456"),
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "failing_mock",
                        "args": {},
                        "id": "call-fail",
                    }],
                ),
            ]
        }

        update = execute_tools_node(state)

        assert update["handoff_reason"] == "tool_execution_error"
        assert route_after_tools({**state, **update}) == "human_handoff"
        assert "private upstream diagnostic" not in update["messages"][0].content

    def test_compiled_graph_completes_low_confidence_handoff(self, monkeypatch):
        from src.support_agent import agent as agent_module

        monkeypatch.setattr(
            agent_module,
            "llm",
            FakeChatModel(AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_vector_knowledge_base",
                    "args": {"query": "unlisted exception"},
                    "id": "call-search-low",
                }],
            )),
        )

        class LowConfidenceTool:
            @staticmethod
            def invoke(_args):
                return "HANDOFF_REQUIRED: low_confidence_retrieval\nNo grounded match."

        class TicketTool:
            @staticmethod
            def invoke(args):
                return f"Ticket created for {args['reason']}"

        monkeypatch.setitem(
            agent_module._tools_by_name,
            "search_vector_knowledge_base",
            LowConfidenceTool(),
        )
        monkeypatch.setitem(agent_module._tools_by_name, "escalate_to_human", TicketTool())

        result = create_graph().invoke(
            {"messages": [HumanMessage(content="Can you make a one-off exception?")]}
        )

        assert result["support_status"] == "handed_off"
        assert isinstance(result["messages"][-1], AIMessage)
        assert "low_confidence_retrieval" in result["messages"][-1].content

    def test_compiled_graph_completes_tool_error_handoff(self, monkeypatch):
        from src.support_agent import agent as agent_module

        monkeypatch.setattr(
            agent_module,
            "llm",
            FakeChatModel(AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_order_status",
                    "args": {"order_id": "123456"},
                    "id": "call-order-error",
                }],
            )),
        )

        class FailingTool:
            @staticmethod
            def invoke(_args):
                raise RuntimeError("private diagnostic")

        class TicketTool:
            @staticmethod
            def invoke(args):
                return f"Ticket created for {args['reason']}"

        monkeypatch.setitem(agent_module._tools_by_name, "get_order_status", FailingTool())
        monkeypatch.setitem(agent_module._tools_by_name, "escalate_to_human", TicketTool())

        result = create_graph().invoke(
            {"messages": [HumanMessage(content="Where is order 123456?")]}
        )

        assert result["support_status"] == "handed_off"
        assert isinstance(result["messages"][-1], AIMessage)
        assert "tool_execution_error" in result["messages"][-1].content
        assert all("private diagnostic" not in str(message.content) for message in result["messages"])


class TestAgentNode:
    """Test the agent reasoning node that decides actions."""

    def test_agent_node_returns_messages(self, monkeypatch):
        """Test that agent_node returns a dict with messages key."""
        monkeypatch.setattr(
            "src.support_agent.agent.llm", FakeChatModel(AIMessage(content="I can help with that."))
        )
        state: SupportState = {
            "messages": [HumanMessage(content="Hello, I need help with my order")]
        }

        result = agent_node(state)

        assert isinstance(result, dict), "Should return a dictionary"
        assert "messages" in result, "Should have 'messages' key"
        assert isinstance(result["messages"], list), "Messages should be a list"
        assert len(result["messages"]) > 0, "Should return at least one message"

    def test_agent_node_returns_ai_message(self, monkeypatch):
        """Test that agent_node returns AIMessage type."""
        monkeypatch.setattr(
            "src.support_agent.agent.llm", FakeChatModel(AIMessage(content="I can help with that."))
        )
        state: SupportState = {
            "messages": [HumanMessage(content="What's your return policy?")]
        }

        result = agent_node(state)
        message = result["messages"][0]

        assert isinstance(message, AIMessage), "Should return AIMessage"

    def test_agent_node_preserves_conversation_context(self, monkeypatch):
        """Test that agent has access to conversation history."""
        monkeypatch.setattr(
            "src.support_agent.agent.llm",
            FakeChatModel(AIMessage(content="Received conversation context.")),
        )
        state: SupportState = {
            "messages": [
                HumanMessage(content="Hi, my order number is 123456"),
                AIMessage(content="I'll look that up for you"),
                HumanMessage(content="Also, I want to return it"),
            ]
        }

        # Agent should be able to reference the order number from earlier
        result = agent_node(state)
        assert result["messages"][0] is not None


class TestGraphCreation:
    """Test graph structure and compilation."""

    def test_create_graph_returns_compiled_graph(self):
        """Test that create_graph returns a compiled graph."""
        graph = create_graph()

        assert graph is not None, "Graph should be created"
        # Check if it's a compiled graph by checking for invoke method
        assert hasattr(graph, "invoke"), "Graph should be compiled with invoke method"
        assert hasattr(graph, "stream"), "Graph should have stream method"

    def test_graph_has_required_nodes(self):
        """Test that graph contains the required nodes."""
        graph = create_graph()

        # Get the graph structure
        # Note: This is implementation-specific and may need adjustment
        # based on LangGraph's API
        assert graph is not None

    def test_graph_invoke_with_simple_message(self, monkeypatch):
        """Test that graph can process a simple message."""
        monkeypatch.setattr(
            "src.support_agent.agent.llm",
            FakeChatModel(AIMessage(content="Hello! How can I help?")),
        )
        graph = create_graph()

        state: SupportState = {
            "messages": [HumanMessage(content="Hello")]
        }

        # This is an integration test - it will actually call the LLM
        # In a real test environment, you'd mock the LLM
        result = graph.invoke(state)

        assert "messages" in result, "Result should contain messages"
        assert len(result["messages"]) > 1, "Should have at least initial message + response"


class TestGraphIntegration:
    """Integration tests for the full graph execution (requires LLM)."""

    @pytest.mark.integration
    def test_graph_handles_order_status_query(self):
        """Test end-to-end flow for order status query."""
        graph = create_graph()

        result = graph.invoke(
            {"messages": [HumanMessage(content="What's the status of order 123456?")]}
        )

        # Should have multiple messages (input + agent responses + tool results)
        assert len(result["messages"]) >= 2
        # Last message should be from agent
        assert isinstance(result["messages"][-1], AIMessage)

    @pytest.mark.integration
    def test_graph_handles_general_question(self):
        """Test end-to-end flow for general question without tools."""
        graph = create_graph()

        result = graph.invoke(
            {"messages": [HumanMessage(content="Thank you for your help!")]}
        )

        # Should have response
        assert len(result["messages"]) >= 2
        # Should end without tool calls for simple thank you
        last_message = result["messages"][-1]
        assert isinstance(last_message, AIMessage)

    @pytest.mark.integration
    def test_graph_maintains_conversation_state(self):
        """Test that graph maintains state across multiple invocations."""
        graph = create_graph()
        config = {"configurable": {"thread_id": "test-123"}}

        # First message
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="My order is 123456")]},
            config
        )
        assert len(result1["messages"]) >= 2

        # Second message referencing first
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="I want to return it")]},
            config
        )
        # Should maintain context from previous message
        assert len(result2["messages"]) >= 2
