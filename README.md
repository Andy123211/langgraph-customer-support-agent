
# Building an Intelligent Customer Support Agent: A Journey Through the LangChain Ecosystem

> **Reproducible demo scope (2026-10):** This repository is a single LangGraph
> customer-support demo. Its policy/FAQ index is process-local
> `InMemoryVectorStore` with BGE-M3 embeddings, BM25 and RRF. A BGE cross-encoder
> reranker is available behind `ENABLE_RERANKER=true`. Order and inventory
> tools use mock data; handoff creates a demo ticket. There is no Milvus,
> calibrated confidence model, intent classifier, long-term memory/PII pipeline,
> Skills runtime, MCP integration, vLLM comparison, or 300-question benchmark in
> this repository. Resume metrics that depend on those systems are **not
> reproduced by this project**.

## Reproducible Local Setup (Windows / Conda)

From the repository root in PowerShell:

```powershell
conda create -n customer_support_resume python=3.12 pip -y
conda activate customer_support_resume
python -m pip install -e ".[dev]" "langgraph-cli[inmem]"
Copy-Item .env.example .env
```

The first semantic search downloads BAAI/bge-m3 weights (about 2.2 GB) and
indexes the bundled JSON policy/FAQ data in memory. Set `HF_HOME` in `.env` to
put model files in a dedicated cache directory. The reranker downloads its own
weights (about 0.6 GB) only when enabled. Both models run on CPU by default.

For local chat inference, install [Ollama](https://ollama.com/download), then in a separate terminal:

```powershell
ollama pull llama3.1:latest
ollama serve
```

Ollama is optional; an OpenAI-compatible API can be configured instead as
described below.

Start the graph API from the activated Conda environment:

```powershell
langgraph dev --no-browser
```

For retrieval and graph handoff tests that do not require model downloads or an
Ollama server:

```powershell
$env:PYTHONNOUSERSITE = "1"
python -m pytest src/support_agent/tests/ -m "not integration"
```

The graph API may start without contacting Ollama; an actual conversation needs
the model pulled and Ollama running. Set `ENABLE_RERANKER=true` in `.env` to
enable BAAI/bge-reranker-v2-m3 after the first model has been downloaded.

### Use an OpenAI-compatible API instead of Ollama

The API format shown in the project setup screenshot can be used by switching
the chat provider. Copy the endpoint and model name into the local `.env`, then
enter the API key locally; the screenshot masks the key, so it is not included
in this repository:

```dotenv
LLM_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_BASE_URL=https://lin312354.top/cliproxy/v1
OPENAI_COMPATIBLE_API_KEY=your_key_here
MODEL_NAME=devin/swe-2(high)
```

Restart `langgraph dev` after changing `.env`. Keep the key private and do not
commit `.env`. The endpoint and selected model must support OpenAI-compatible
Chat Completions with tool calls, since the agent binds its retrieval and order
tools to the chat model. This mode sends prompts to that external API and may
incur usage charges. Ollama remains the default local option when
`LLM_PROVIDER` is unset or set to `ollama`.

The benchmark claims in the resume (300 questions, Recall@5/MRR, refusal rate,
Youden-J threshold, classifier F1 and ONNX parity) require source datasets and
artifacts that are not present here. The existing LangSmith script uses 10
examples and requires user-provided LangSmith credentials; it does not
reproduce those resume metrics.

## Introduction

Imagine a customer service agent that never sleeps, responds instantly to inquiries about orders and refunds, knows when to escalate complex issues to humans, and continuously improves through rigorous evaluation. This project brings that vision to life as a demonstration of the LangChain ecosystem's capabilities for building production-ready AI agents.

At its heart, this is a story about learning by doing—exploring the vast landscape of agent architectures, tooling, and evaluation frameworks while keeping one principle paramount: **simplicity first**.

## Design Philosophy: Keep It Simple

Rather than building a sprawling multi-agent system from the outset, this project embraces an intentionally straightforward architecture. The customer support agent serves as a focused exploration of core concepts: tool usage, knowledge retrieval, human-in-the-loop patterns, and comprehensive testing. By mastering these fundamentals, we create a solid foundation for future complexity.

### The Agent's Journey

When a customer reaches out, the agent searches an in-memory index of store policies and FAQs, then consults mock order or inventory data when appropriate. If retrieval is weak or a tool fails, LangGraph routes the request to a demo human-support ticket.

![Agent Workflow Graph](images/langgraph-graph-mermaid.png)

*The agent's workflow: a clean cycle between reasoning, tool usage, and knowing when to stop or escalate.*

### Architecture in Action

The system architecture reflects real-world production considerations:

- **Knowledge Layer**: Process-local vector index for policy documents and mock customer data stores
- **Intelligence Layer**: Powered by Ollama's `llama3.1:latest` model for local, cost-effective inference
- **Interface Layer**: LangChain Studio and Agent UI provide a streaming chat experience
- **Quality Assurance**: Comprehensive TDD tests ensure reliability
- **Deployment**: Docker configuration for consistent environments across development and production

![Agent Chat Interface](images/agent-ui-start.png)

![Agent UI Greeting](images/agent-ui-greeting.png)

![Agent UI Tool List](images/agent-ui-tool-list.png)

![Agent UI Orders](images/agent-ui-orders.png)

*The Agent Chat UI: a simple, clean interface for interacting with the customer support agent via localhost deployment.*

### Rigorous Evaluation Framework

Quality isn't accidental—it's measured. The evaluation system puts the agent through its paces with 10 diverse customer scenarios, scoring responses across five dimensions:

1. **Tool Usage**: Does the agent call the right functions?
2. **Keyword Presence**: Are critical terms and information included?
3. **Response Quality**: Is the answer accurate and complete?
4. **Empathy**: Does the agent acknowledge customer concerns?
5. **Actionability**: Can the customer act on the guidance provided?

Every evaluation run uploads detailed results to the [LangSmith dashboard](https://smith.langchain.com), complete with conversation traces, per-test feedback explaining pass/fail decisions, and aggregate scores. This creates a feedback loop for continuous improvement and regression detection.

![LangSmith Test Examples](images/eval-summary-examples.png)

*The evaluation dataset: 10 diverse customer scenarios testing everything from frustrated complaints to simple inventory questions.*

![LangSmith Evaluation Metrics](images/eval-summary-experiments.png)

*Performance metrics across all evaluation dimensions, with aggregate scores showing how the agent performs on tool usage, response quality, empathy, and more.*

**Running Evaluations**: Execute `python -m src.support_agent.tests.eval_langsmith` and view comprehensive metrics in LangSmith's web interface to track performance trends over time.

## The Road Ahead

This foundation opens doors to exciting possibilities:

**Live Customer Data**: Implement it to read/edit a database of customer data - postgres or SQLite.

**Multi-Agent Workflows**: Evolve from a single agent to a supervised ensemble where specialized agents handle distinct business processes—one for refunds, another for order modifications, each an expert in its domain. A primary assistant would orchestrate these specialists, routing requests to the right team member.

**Enhanced Human Collaboration**: Implement more sophisticated interrupt patterns for confirmations and executions, creating seamless handoffs between AI and human agents.

**Enterprise Data Integration**: Replace mock data with SQL-based customer databases, connecting the agent to real business systems and workflows.

The journey from simple to sophisticated is just beginning—but the foundation is rock solid.

For detailed developer setup instructions, see [DEVELOPER_SETUP.md](DEVELOPER_SETUP.md).

# Customer Support Bot Technical Documentation

A customer-support agent demo built with LangGraph, featuring tool calling, conversation history, hybrid knowledge retrieval, and local LLM support via Ollama.

## 🎉 What You Have

A runnable customer-support demo built with:

- ✅ LangGraph for orchestration
- ✅ Local LLM via Ollama (privacy-first)
- ✅ 5 pre-built tools (order tracking, returns, KB search, etc.)
- ✅ Visual debugging with LangGraph Studio
- ✅ REST API ready to use
- ✅ Full documentation and examples
- ✅ Hybrid BM25 + vector retrieval with reciprocal-rank fusion (RRF)
- ✅ Low-relevance and tool-error handoff state transitions
- ✅ LangSmith evaluation suite with online dashboard
- ✅ Pytest suite for graph routing, tools, state, and retrieval components

## Features

- 🤖 **Intelligent Agent**: Uses ReAct pattern with tool calling
- 💬 **Multi-turn Conversations**: Built-in memory
- 🔧 **5 Support Tools**: Knowledge base, order tracking, returns, inventory, escalation
- 🎨 **LangGraph Studio**: Visual debugging and testing UI
- 🚀 **Local First**: Runs entirely on your machine with Ollama
- 📊 **REST API**: Auto-generated endpoints via LangGraph Dev
- 🔍 **Hybrid RAG**: BM25 and vector recall fused with RRF; an optional reranker adapter can reorder fused candidates
- 📊 **LangSmith Evals**: Automated testing with online dashboard

### Hybrid retrieval and handoff behavior

Policy and product questions use two retrieval paths over the same local knowledge base:

1. BM25 ranks exact terms and identifiers.
2. `InMemoryVectorStore` ranks semantic matches with the configured Hugging Face embedding model.
3. Reciprocal-rank fusion combines the two ranked lists. Set `ENABLE_RERANKER=true` to load the local BAAI/bge-reranker-v2-m3 cross-encoder and rerank fused candidates.
4. The tool reports a bounded relevance heuristic. If no result reaches the default `0.35` threshold, the graph records `handoff_required` and creates a human-support ticket instead of answering from weak evidence.

The threshold and displayed score are demo heuristics, not calibrated confidence probabilities. Tune them against a held-out support-query set before using this pattern beyond a demo. Order, return, and inventory tools remain mock-data examples. Tool exceptions are caught by the graph, hidden from the customer, and routed to the human handoff node.

## Prerequisites

- Python 3.11-3.12 (3.12 recommended)
- [Ollama](https://ollama.ai) installed and running
- Node.js/pnpm (optional, for TypeScript projects)

## 🚀 Get Started in 3 Steps

### Step 1: Install Ollama (2 minutes)

**macOS:**

```bash
brew install ollama
ollama serve
```

**Linux:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

**Windows:**
Download from <https://ollama.ai/download/windows>

**Pull the model:**

```bash
ollama pull llama3.1:latest
```

### Step 2: Setup Python Environment and Environment Variables (2 minutes)

```bash
cd langgraph-customer-support-agent

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
pip install langgraph-cli
```

**Configure Environment Variables:**

Create a `.env` file in the project root with the following variables:

```bash
# Required: Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Required: Model Configuration
MODEL_NAME=llama3.1:latest

# Required for LangSmith evaluation and tracing
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=focused-io-test
LANGCHAIN_TRACING_V2=true

# LangGraph Configuration
LANGGRAPH_URL=http://localhost:2024
```

**Get your LangSmith API key:**

1. Sign up at [https://smith.langchain.com](https://smith.langchain.com)
2. Go to Settings → API Keys
3. Copy your API key and replace `your_api_key_here` above

**Note:** If you're using Docker, these environment variables can also be set in `docker-compose.yml` or passed via `.env` file (Docker Compose automatically loads `.env` files).

### Step 3: Run It! (30 seconds)

**Option A: LangGraph Studio (Recommended)**

```bash
langgraph dev
```

Then open: <http://127.0.0.1:2024>

**Option B: Agent UI Interface**

The Agent UI provides an interactive chat interface for testing your agent. To use it:

1. Start the LangGraph server:

```bash
langgraph dev
```

2. Open the Agent UI in your browser:

```
http://localhost:2024
```

3. Click "Start Agent" to begin:

![Agent UI Start](images/agent-ui-start.png)

![Agent UI Greeting](images/agent-ui-greeting.png)

![Agent UI Tool List](images/agent-ui-tool-list.png)

![Agent UI Orders](images/agent-ui-orders.png)

4. Enter your messages in the chat interface and interact with the customer support bot
5. The UI shows tool calls and agent responses in real-time

**Option C: Command Line**

```bash
python scripts/cli.py
```

**Option D: Run Tests**

```bash
python scripts/test_bot.py
```

## ⚙️ Environment Variables

Before running the agent, you need to configure environment variables. Create a `.env` file in the project root:

```bash
# Required: Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Required: Model Configuration
MODEL_NAME=llama3.1:latest

# Required for LangSmith evaluation and tracing
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=focused-io-test
LANGCHAIN_TRACING_V2=true

# LangGraph Configuration
LANGGRAPH_URL=http://localhost:2024
```

### Environment Variable Reference

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Yes | URL where Ollama service is running | `http://localhost:11434` |
| `MODEL_NAME` | Yes | The Ollama model to use | `llama3.1:latest` |
| `LANGSMITH_ENDPOINT` | Yes* | LangSmith API endpoint | `https://api.smith.langchain.com` |
| `LANGSMITH_API_KEY` | Yes* | Your LangSmith API key | None |
| `LANGSMITH_PROJECT` | Yes* | LangSmith project name | `focused-io-test` |
| `LANGCHAIN_TRACING_V2` | Yes* | Enable tracing to LangSmith | `true` |
| `LANGGRAPH_URL` | Yes | LangGraph URL | `http://localhost:2024` |

*Required for LangSmith evaluation and tracing features. Can be omitted if not using those features.

**Getting your LangSmith API key:**

1. Sign up at [https://smith.langchain.com](https://smith.langchain.com)
2. Go to Settings → API Keys
3. Copy your API key and replace `your_api_key_here` in your `.env` file

## 📁 Project Structure

```
langgraph-customer-support-agent/
├── 📖 Documentation
│   ├── README.md                  # This file - complete documentation
│   ├── DEVELOPER_SETUP.md         # Developer setup guide
│   └── .gitignore                # Git ignore patterns
│
├── 🐍 Python Application
│   ├── src/support_agent/         ⚙️ Main application code
│   │   ├── __init__.py           
│   │   ├── agent.py              🎯 CORE: Graph definition
│   │   ├── tools.py              🔧 5 support tools
│   │   ├── state.py              💾 State schema
│   │   ├── prompts.py            💬 System instructions
│   │   ├── vector_store.py       🔍 Vector search implementation
│   │   └── tests/                🧪 Test suite and evaluations
│   │       ├── test_agent.py     # Agent graph logic tests
│   │       ├── test_tools.py     # Support tools tests
│   │       ├── test_state.py     # State management tests
│   │       ├── eval_langsmith.py # LangSmith evaluation script
│   │       ├── custom_evaluators.py # Custom evaluators
│   │       └── README.md         # Test documentation
│   │
│   └── scripts/                   🎬 Ready-to-run scripts
│       ├── cli.py                 💻 Interactive chat
│       ├── test_bot.py            🧪 Test suite
│       ├── api_client_example.py 📡 API examples
│       └── init_vector_store.py  🔍 Vector store testing
│
├── 📊 Data
│   └── data/
│       └── knowledge_base.json    📚 Mock KB data
│
├── ⚙️ Configuration
│   ├── langgraph.json             🔗 LangGraph config (required!)
│   ├── pyproject.toml             📦 Project metadata
│   ├── requirements.txt            📋 Dependencies
│   └── .env.example               🔐 Environment template
│
├── 🐳 Docker (Optional)
│   ├── Dockerfile                 📦 Container image
│   └── docker-compose.yml         🎼 Multi-container setup
│
└── 📁 Storage (Generated at Runtime)
    └── storage/                   # Created automatically
        └── checkpoints.db         # SQLite database for conversation memory
```

### Key Files

**agent.py** - The heart of the application

- Defines the LangGraph workflow
- Creates agent and tool nodes
- Implements ReAct loop (reasoning + acting)
- Exports `graph` variable for LangGraph Dev

**tools.py** - Business logic

- `search_knowledge_base()` - Find policies and FAQ using vector search
- `get_order_status()` - Look up orders
- `initiate_return()` - Process returns
- `check_product_availability()` - Check inventory
- `escalate_to_human()` - Create support tickets

**state.py** - Data schema

- Defines `SupportState` TypedDict
- Manages conversation history with `add_messages`

**prompts.py** - Agent instructions

- System prompt that guides agent behavior
- Defines when to use tools
- Sets tone and guidelines

**vector_store.py** - Semantic search

- Implements vector-based knowledge base search
- Uses HuggingFace embeddings for semantic understanding
- Provides intelligent document retrieval

## 🎯 Try These First

Once running, try these example queries:

1. "What's your return policy?"
2. "Can you check order #123456?"
3. "I want to return my order because it's defective"
4. "Do you have laptops in stock?"
5. "This is terrible service!" (watch it escalate!)

## Usage Examples

### Using the Agent UI Interface

The Agent UI provides an interactive web interface to chat with your customer support bot:

1. **Start the LangGraph server**:

   ```bash
   langgraph dev
   ```

2. **Open the Agent UI** in your browser:

   ```
   http://localhost:2024
   ```

3. **Start the conversation**:

   ![Agent UI Start](images/agent-ui-start.png)

   ![Agent UI Greeting](images/agent-ui-greeting.png)

   ![Agent UI Tool List](images/agent-ui-tool-list.png)

   ![Agent UI Orders](images/agent-ui-orders.png)

   Click "Start Agent" or begin typing your message in the chat interface

4. **Interact with the bot**:
   - Type customer queries in the chat input
   - Watch the agent respond in real-time
   - See tool calls as they happen
   - View the conversation history

5. **Features of the UI**:
   - Real-time streaming responses
   - Tool call visibility
   - Conversation persistence
   - Clean, intuitive interface

### Using LangGraph Studio (Advanced Debugging)

For advanced debugging and visualization:

1. Start server: `langgraph dev`
2. Open <http://127.0.0.1:2024>
3. Type messages in the chat interface
4. Watch tool calls happen in real-time
5. Inspect state at each step
6. Set breakpoints and time-travel debug

### Using CLI

```bash
python scripts/cli.py

# Example conversation:
You: Hi, what's your return policy?
🤖 Bot: We accept returns within 30 days of delivery...

You: Can you check order #123456?
🔧 Using tool: get_order_status...
🤖 Bot: Order #123456 is in transit...
```

### Using the API

```python
import requests

# Create thread
response = requests.post("http://localhost:2024/threads")
thread_id = response.json()["thread_id"]

# Send message
requests.post(
    f"http://localhost:2024/threads/{thread_id}/runs",
    json={
        "input": {
            "messages": [{"role": "user", "content": "What's your return policy?"}]
        }
    }
)
```

## Available Tools

The bot has access to these tools:

1. **search_knowledge_base** - Find store policies and FAQ using semantic search
2. **get_order_status** - Look up order tracking
3. **initiate_return** - Start return process
4. **check_product_availability** - Check stock levels
5. **escalate_to_human** - Transfer to human agent

## 🧪 Testing

The pytest suite contains deterministic tests for retrieval, graph routing, state, and mock business tools. It does not include the resume's external benchmark datasets.

### Test Structure

```
src/support_agent/tests/
├── test_agent.py       # Agent graph logic (10 tests)
├── test_tools.py       # Support tools (26 tests)
├── test_state.py       # State management (14 tests)
├── conftest.py         # Pytest fixtures
├── eval_langsmith.py   # LangSmith evaluation script
└── custom_evaluators.py # Custom evaluators (7 available)
```

### Running Tests

**IMPORTANT**: Activate the virtual environment first!

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Run all unit tests (fast, no LLM required)
pytest src/support_agent/tests/ -m "not integration"

# Run all tests including integration tests (requires Ollama)
pytest src/support_agent/tests/

# Run with verbose output
pytest src/support_agent/tests/ -v

# Run specific test file
pytest src/support_agent/tests/test_tools.py

# Run specific test
pytest src/support_agent/tests/test_tools.py::TestGetOrderStatus::test_get_order_status_in_transit
```

### Quick Test Script

Use the provided test runner script:

```bash
./scripts/run_tests.sh
```

### Test Coverage

- ✅ **Agent Logic**: Conditional routing, message handling, graph structure
- ✅ **Tools**: Order status, returns, product availability, escalation
- ✅ **State Management**: Message accumulation, type preservation, conversation flow
- ✅ **Mock Data**: Data integrity and consistency

### Example Unit Tests

```python
# test_tools.py - Test order status lookup
def test_get_order_status_in_transit():
    result = get_order_status.invoke({"order_id": "123456"})
    assert "123456" in result
    assert "in_transit" in result.lower()
    assert "tracking" in result.lower()

# test_agent.py - Test conditional routing
def test_should_continue_with_tool_calls():
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "get_order_status"}])
        ]
    }
    result = should_continue(state)
    assert result == "tools"
```

### Integration Tests

Integration tests marked with `@pytest.mark.integration` test end-to-end flows and require:

- Ollama running with llama3.1:latest model
- Longer execution time

Skip integration tests for quick development:

```bash
pytest src/support_agent/tests/ -m "not integration"
```

### Legacy Test Scripts

```bash
# Run interactive test conversations
python scripts/test_bot.py

# Test vector store functionality
python scripts/init_vector_store.py
```

### Writing New Tests

Follow TDD principles when adding features:

1. **Write test first** - Define expected behavior
2. **Run test** - Verify it fails (red)
3. **Implement feature** - Make it work
4. **Run test again** - Verify it passes (green)
5. **Refactor** - Clean up while keeping tests green

## 🎯 LangSmith Evaluations

Evaluate your agent's performance with automated testing that shows up in an online dashboard.

### Quick Start

1. **Setup LangSmith**:

   ```bash
   export LANGCHAIN_API_KEY="your-key-from-smith.langchain.com"
   export LANGCHAIN_TRACING_V2=true
   ```

2. **Check Setup**:

   ```bash
   python scripts/check_langsmith_setup.py
   ```

3. **Run Evaluation**:

   ```bash
   python -m src.support_agent.tests.eval_langsmith
   ```

### What You Get

- **Dataset**: 10 test cases covering various customer scenarios
- **Evaluators**: 5 automated metrics (tool usage, keyword presence, response quality, empathy, actionability)
- **Online Dashboard**: View results at <https://smith.langchain.com>
- **Traces**: Detailed execution traces for debugging

Example output:

```
Average Scores:
  - tool_usage          : 90.00%
  - keyword_presence    : 85.00%
  - response_quality    : 100.00%
  - empathy             : 80.00%
  - actionability       : 95.00%

View results: https://smith.langchain.com/projects
```

### Available Evaluators

The evaluation system includes **5 active evaluators** and **7 additional custom evaluators**:

#### Currently Active (5 evaluators)

1. ✅ **Tool Usage** - Checks if correct tool was called
2. ✅ **Keyword Presence** - Verifies expected info in response
3. ✅ **Response Quality** - Checks completeness and formatting
4. ✅ **Empathy** - Detects empathetic language in difficult situations
5. ✅ **Actionability** - Ensures response provides clear next steps

#### Ready to Activate (7 more)

6. **Politeness** - Checks for courtesy and professional tone
7. **Conciseness** - Evaluates appropriate response length
8. **Specificity** - Prefers specific details over generic statements
9. **Tool Efficiency** - Checks for efficient tool usage
10. **Resolution Clarity** - Ensures clear problem resolution

### Adding More Evaluators

Edit `src/support_agent/tests/eval_langsmith.py`:

```python
from src.support_agent.tests.custom_evaluators import (
    empathy_evaluator,
    actionability_evaluator,
    politeness_evaluator,      # NEW
    conciseness_evaluator,     # NEW
)

evaluators.extend([
    empathy_evaluator,
    actionability_evaluator,
    politeness_evaluator,      # Add more
    conciseness_evaluator,
])
```

### Viewing Results

After running the evaluation, go to <https://smith.langchain.com> and search for "support-agent-eval" to see:

- **Aggregate scores** - Average for each evaluator
- **Per-example results** - Scores for each test case
- **Feedback comments** - Why each test passed/failed
- **Full traces** - Complete conversation flows

### Test Cases

The evaluation includes 10 diverse test cases:

| # | Category | Question |
|---|----------|----------|
| 1 | return_policy | What's your return policy? |
| 2 | shipping | Do you offer international shipping? |
| 3 | payment | What payment methods do you accept? |
| 4 | order_status | Can you check the status of order #123456? |
| 5 | order_status | Where is my order 789012? |
| 6 | return_request | I want to return order #123456 because it's defective |
| 7 | inventory | Is the wireless mouse in stock? |
| 8 | inventory | Do you have any laptops available? |
| 9 | complex | Hi! I ordered a keyboard last week... |
| 10 | escalation | This is ridiculous! I've been waiting for weeks... |

### Creating Custom Evaluators

You can create your own evaluators:

```python
from langsmith.schemas import Run, Example

def my_evaluator(run: Run, example: Example) -> dict:
    """Description of what this checks."""
    response = run.outputs.get("answer", "")

    # Your logic here
    if some_condition:
        return {
            "key": "my_metric",
            "score": 1.0,
            "comment": "Why it passed"
        }
    else:
        return {
            "key": "my_metric",
            "score": 0.0,
            "comment": "Why it failed"
        }
```

See `src/support_agent/tests/custom_evaluators.py` for implementation examples and detailed comments.

## 🔧 Customization

### Adding a New Tool

Tools are how the agent interacts with external systems. Here's how to add one:

#### 1. Define the Tool

Edit `src/support_agent/tools.py`:

```python
@tool
def your_new_tool(param1: str, param2: int = 0) -> str:
    """Short description of what the tool does.
    
    Use this tool when:
    - Situation 1
    - Situation 2
    
    Args:
        param1: Description of parameter
        param2: Optional parameter with default
    
    Returns:
        Description of what the tool returns
    """
    # Your implementation here
    result = do_something(param1, param2)
    return f"Result: {result}"
```

#### 2. Add to Tools List

At the bottom of `tools.py`:

```python
tools = [
    search_knowledge_base,
    get_order_status,
    # ... existing tools
    your_new_tool,  # Add here
]
```

#### 3. Test It

```python
python scripts/test_bot.py
```

That's it! The agent will automatically learn to use your tool.

### Modifying the System Prompt

The system prompt guides the agent's behavior.

Edit `src/support_agent/prompts.py`:

```python
SYSTEM_PROMPT = """You are a helpful customer support agent...

Your job is to:
1. [Add your custom instructions]
2. [More instructions]
...
"""
```

### Changing the LLM Model

Edit `src/support_agent/agent.py`:

```python
llm = ChatOllama(
    model="llama3.1:latest",  # Change this
    temperature=0.3,      # Adjust creativity (0-1)
)
```

Available Ollama models:

- `llama3.1:latest` - Fast, efficient (recommended)
- `mistral:7b` - Good balance
- `qwen2.5:7b` - Strong reasoning
- `llama3.1:8b` - Larger, more capable

### Adding RAG (Vector Search)

The project already includes vector search implementation! To customize:

#### 1. Modify Vector Store

Edit `src/support_agent/vector_store.py`:

```python
# Use different embedding model
def __init__(self, embeddings_model: str = "sentence-transformers/all-mpnet-base-v2"):
```

Popular alternatives:

- `all-mpnet-base-v2` - Higher quality, slower
- `all-MiniLM-L12-v2` - Balance between speed and quality

#### 2. Test Vector Search

```bash
python scripts/init_vector_store.py
```

### Integrating Real Systems

#### Connect to Order Management System

```python
@tool
def get_order_status(order_id: str) -> str:
    """Look up real order status."""
    import requests
    
    response = requests.get(
        f"https://your-api.com/orders/{order_id}",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    
    order = response.json()
    return format_order_status(order)
```

#### Connect to CRM

```python
@tool
def get_customer_info(email: str) -> str:
    """Get customer details from CRM."""
    # Connect to Salesforce, HubSpot, etc.
    customer = crm_client.get_customer(email)
    return format_customer_info(customer)
```

### Adding Conversation Memory

LangGraph already includes checkpointing, but you can extend it:

```python
# In agent.py
from langgraph.checkpoint.postgres import PostgresSaver

# Use PostgreSQL instead of SQLite
memory = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/dbname"
)
```

## 🐳 Docker Setup (Recommended for Quick Start)

The easiest way to get started! Everything is pre-configured and ready to run.

### Quick Start with Docker

**Prerequisites:**

- Docker and Docker Compose installed ([Get Docker](https://docs.docker.com/get-docker/))
- That's it! No need to install Python, Ollama, or any dependencies manually

**Start Everything:**

```bash
# Clone the repository (if you haven't already)
git clone <repository-url>
cd langgraph-customer-support-agent

# Start all services (first run will take 5-10 minutes to download models)
docker-compose up
```

**What happens automatically:**

1. ✅ Ollama service starts
2. ✅ Downloads llama3.1:latest model (happens once, ~2GB)
3. ✅ Support bot starts and connects to Ollama
4. ✅ API ready at <http://localhost:2024>

**Access the application:**

- Agent UI Interface: <http://localhost:2024>
- Ollama API: <http://localhost:11434>

### Using Agent UI with Docker

Once Docker is running, access the Agent UI interface:

1. **Start the Docker containers**:

   ```bash
   docker-compose up
   ```

   Wait for the message: "Ollama initialization complete!"

2. **Access the Agent UI** in your browser:

   ```
   http://localhost:2024
   ```

3. **Start chatting**:

   ![Agent UI Start](images/agent-ui-start.png)

   ![Agent UI Greeting](images/agent-ui-greeting.png)

   ![Agent UI Tool List](images/agent-ui-tool-list.png)

   ![Agent UI Orders](images/agent-ui-orders.png)

   Click "Start Agent" and begin interacting with your customer support bot

### Docker Commands

```bash
# Start in background (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f support-bot
docker-compose logs -f ollama
docker-compose logs ollama-init

# Stop all services
docker-compose down

# Stop and remove all data (including downloaded models)
docker-compose down -v

# Rebuild after code changes
docker-compose up --build

# Use a different model
OLLAMA_MODEL=llama3.1:latest docker-compose up
```

### Docker Architecture

The setup includes three services:

1. **ollama** - Ollama service for running LLMs
   - Automatically starts and stays healthy
   - Models stored in persistent volume
   - Accessible at <http://ollama:11434> (internal) and <http://localhost:11434> (external)

2. **ollama-init** - One-time model initialization
   - Automatically pulls llama3.1:latest model on first run
   - Ensures model is ready before starting the bot
   - Exits after successful initialization

3. **support-bot** - The LangGraph customer support agent
   - Starts only after Ollama and model are ready
   - Hot-reload enabled for development (code changes auto-reload)
   - Exposes API on port 2024

### Benefits of Docker Setup

✅ **Zero Configuration** - Everything just works out of the box
✅ **Consistent Environment** - Same setup on any OS (Mac, Linux, Windows)
✅ **Isolated** - Won't conflict with your system Python/Ollama installations
✅ **Easy Cleanup** - Remove everything with one command
✅ **Production Ready** - Same setup works for dev and production
✅ **Hot Reload** - Code changes automatically reload in the container

The Docker setup automatically configures:

- LangGraph server on port 2024
- Ollama service with llama3.1:latest model pre-loaded
- Persistent storage for conversation history
- Network connectivity between all services
- Volume mounts for live code editing

### Customizing the Docker Setup

**Change the model:**

Edit [docker-compose.yml](docker-compose.yml):

```yaml
ollama-init:
  environment:
    - OLLAMA_HOST=http://ollama:11434
    - OLLAMA_MODEL=llama3.1:latest  # Change this line
```

Available models:

- `llama3.1:latest` - Recommended, fast, efficient (~1GB RAM)
- `mistral:7b` - More capable (~5GB RAM)
- `qwen2.5:7b` - Strong reasoning (~5GB RAM)
- `llama3.1:8b` - Larger, most capable (~7GB RAM)

**Development mode with hot reload:**

The docker-compose.yml already mounts source directories:

```yaml
volumes:
  - ./src:/app/src        # Code changes reload automatically
  - ./data:/app/data      # Knowledge base changes
  - ./storage:/app/storage # Persistent conversation history
```

Just edit your code locally and it will reload automatically in the container!

### Troubleshooting Docker

**Issue: Slow first startup**

- This is normal! First run downloads the Ollama model (~2GB)
- Subsequent starts are much faster (5-10 seconds)
- Check progress: `docker-compose logs -f ollama-init`

**Issue: Port already in use**

```bash
# Change ports in docker-compose.yml
services:
  support-bot:
    ports:
      - "8124:2024"  # Use port 8124 instead of 2024
```

**Issue: Out of memory**

- Ollama models need 4-8GB RAM available
- Free up memory or use a smaller model (llama3.1:latest)
- Check Docker Desktop settings to allocate more RAM

**Issue: Model download fails**

```bash
# Check internet connection and logs
docker-compose logs ollama-init

# Manually pull model
docker-compose exec ollama ollama pull llama3.1:latest

# Restart services
docker-compose restart
```

**Issue: Container won't start**

```bash
# Check container status
docker-compose ps

# View detailed logs
docker-compose logs

# Clean restart
docker-compose down
docker-compose up --build
```

**Issue: Changes not reflecting**

```bash
# For Dockerfile changes, force rebuild:
docker-compose down
docker-compose up --build

# For Python code changes, just restart the service:
docker-compose restart support-bot

# Note: Volume-mounted files (src/, data/) should auto-reload
```

**Issue: Can't connect to Ollama**

```bash
# Check if Ollama service is healthy
docker-compose ps

# Test Ollama directly
curl http://localhost:11434/api/tags

# Check network connectivity between containers
docker-compose exec support-bot curl http://ollama:11434/api/tags
```

## 🆘 Troubleshooting

### Installation Issues

#### Python Version Error

**Error**: `requires-python = ">=3.11,<3.13"`

**Solution**:

```bash
# Check your Python version
python --version

# Must be Python 3.11 or 3.12 (3.12 recommended)
# If version is < 3.11 or >= 3.13, install Python 3.12
# macOS:
brew install python@3.12

# Linux:
sudo apt install python3.12

# Windows: Download Python 3.12 from python.org
```

#### ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'langgraph'`

**Solution**:

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### Import Error from 'src'

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:

```bash
# Make sure you're in the project root
pwd  # Should show: .../langgraph-customer-support-agent

# Install package in editable mode
pip install -e .

# Or run scripts with full path
python -m scripts.cli
```

### Ollama Issues

#### Ollama Not Running

**Error**: `Connection refused` or `Failed to connect to Ollama`

**Solution**:

```bash
# Check if Ollama is running
ollama list

# If not running, start it
ollama serve

# On macOS, you can also:
brew services start ollama
```

#### Model Not Found

**Error**: `Model 'llama3.1:latest' not found`

**Solution**:

```bash
# Pull the model
ollama pull llama3.1:latest

# Verify it's installed
ollama list

# If you want a different model
ollama pull llama3.1:latest
# Then update src/support_agent/agent.py
```

#### Ollama Running but Not Responding

**Solution**:

```bash
# Check Ollama logs
tail -f ~/.ollama/logs/server.log

# Restart Ollama
pkill ollama
ollama serve

# Test Ollama directly
curl http://localhost:11434/api/tags
```

### LangGraph Dev Issues

#### Port Already in Use

**Error**: `Address already in use: 2024`

**Solution**:

```bash
# Option 1: Use different port
langgraph dev --port 8124

# Option 2: Find and kill process using port 2024
# macOS/Linux:
lsof -ti:2024 | xargs kill -9

# Windows:
netstat -ano | findstr :2024
taskkill /PID <PID> /F
```

#### Graph Not Found

**Error**: `Graph 'support_agent' not found`

**Solution**:

```bash
# Verify langgraph.json is correct
cat langgraph.json

# Make sure file paths are correct
# Should be: "./src/support_agent/agent.py:graph"

# Try absolute import
python -c "from src.support_agent.agent import graph; print(graph)"
```

#### Changes Not Reflecting

**Issue**: Code changes not appearing in LangGraph Studio

**Solution**:

```bash
# LangGraph Dev auto-reloads, but if stuck:
# 1. Stop server (Ctrl+C)
# 2. Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +
# 3. Restart
langgraph dev
```

### Runtime Issues

#### Agent Not Using Tools

**Issue**: Bot responds without calling tools

**Possible Causes**:

1. **Ollama model doesn't support tool calling well**
   - Solution: Use llama3.1:latest, mistral:7b, or qwen2.5:7b
   - These models have better tool calling support

2. **System prompt unclear**
   - Solution: Make tool usage more explicit in prompts.py

3. **Tools not properly bound**
   - Check: `llm.bind_tools(tools)` in agent.py

**Debug**:

```python
# Test tool calling directly
from src.support_agent.agent import llm
result = llm.invoke("Use get_order_status for order #123456")
print(result.tool_calls)  # Should not be empty
```

#### Slow Responses

**Issue**: Bot takes too long to respond

**Solutions**:

```bash
# 1. Use faster model
# Edit src/support_agent/agent.py
llm = ChatOllama(model="llama3.1:latest")  # Fastest

# 2. Check Ollama performance
ollama run llama3.1:latest "Hello"  # Should be fast

# 3. Reduce context
# Limit conversation history if too long

# 4. Check system resources
htop  # macOS/Linux
# Task Manager (Windows)
# Ollama needs RAM for models
```

#### Memory/State Issues

**Issue**: Bot doesn't remember previous messages

**Solution**:

```python
# Check if checkpointer is configured
# In agent.py, verify:
memory = SqliteSaver.from_conn_string("...")
graph = workflow.compile(checkpointer=memory)

# Check if thread_id is consistent
# In CLI/API, make sure thread_id stays the same

# Verify checkpoint file
ls -lh storage/checkpoints.db
```

#### Tool Execution Errors

**Issue**: Tools failing silently

**Debug**:

```bash
# Run with more logging
export LANGCHAIN_VERBOSE=true
python scripts/cli.py

# Or add print statements in tools.py
@tool
def my_tool(param: str) -> str:
    print(f"DEBUG: Tool called with {param}")
    # ... rest of tool
```

### API Issues

#### API Returns 404

**Error**: `404 Not Found`

**Solution**:

```bash
# Check base URL
curl http://localhost:2024/

# Verify thread exists before sending message
# Create thread first:
curl -X POST http://localhost:2024/threads

# Check LangGraph Dev is running
ps aux | grep langgraph
```

#### API Returns 500

**Error**: `500 Internal Server Error`

**Solution**:

```bash
# Check LangGraph Dev logs
# They print to terminal where you ran 'langgraph dev'

# Common causes:
# 1. Ollama not running
# 2. Model not loaded
# 3. Tool error

# Test with curl
curl -X POST http://localhost:2024/threads/test/runs \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "test"}]}}'
```

#### Streaming Not Working

**Issue**: Can't stream responses

**Solution**:

```python
# Make sure to set stream_mode
payload = {
    "input": {"messages": [...]},
    "stream_mode": "values"  # or "updates"
}

# Use requests with stream=True
response = requests.post(url, json=payload, stream=True)
for line in response.iter_lines():
    # Process line
```

## Performance Optimization

### Reduce Memory Usage

```python
# Use smaller model
llm = ChatOllama(model="llama3.1:latest")

# Limit conversation history
def agent_node(state: SupportState) -> dict:
    # Keep only last 10 messages
    messages = state["messages"][-10:]
    # ... rest of function
```

### Speed Up Tool Execution

```python
# Use async tools
@tool
async def fast_tool(param: str) -> str:
    """Async tool."""
    result = await async_operation(param)
    return result

# Cache results
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_lookup(key: str) -> str:
    # Expensive operation
    return result
```

## 🔍 Vector Store Implementation

The knowledge base uses semantic vector search for intelligent document retrieval.

### How It Works

1. **Document Creation**: JSON knowledge base is converted into LangChain `Document` objects with:
   - `page_content`: The actual text content
   - `metadata`: Category tags for filtering (return, shipping, payment, product, general)

2. **Embedding Generation**: Each document is converted to a vector embedding using the configured model (default: `BAAI/bge-m3`)

3. **Vector Search**: User queries are:
   - Converted to embeddings
   - Compared to all document embeddings using cosine similarity
   - Top-k most similar documents are returned
   - Optional category filtering narrows results

### Benefits

- **Semantic Understanding**: Finds relevant results even when exact keywords don't match
- **Better User Experience**: More intelligent responses to customer queries
- **Flexibility**: Easy to add new documents without restructuring code
- **Performance**: In-memory storage provides fast lookups
- **Scalability**: Can easily swap to persistent vector stores (ChromaDB, Pinecone, etc.) for production

### Usage

```python
# General search (all categories)
results = vector_store.search("Can I get a refund?", k=3)

# Category-specific search
results = vector_store.search("shipping options", k=2, filter_category="shipping")
```

### Testing Vector Store

```bash
python scripts/init_vector_store.py
```

## Development

### Hot Reload

LangGraph Dev automatically reloads when you change code. Just save your files and test immediately.

### Graph Visualization

Visualize the LangGraph workflow structure:

**LangGraph Workflow Diagram:**
![LangGraph Customer Support Agent Workflow](images/langgraph-graph-mermaid.png)

The diagram above shows the complete workflow of the customer support agent:

- **Start** → **Agent**: Initial conversation begins
- **Agent** → **End**: Agent can conclude the conversation
- **Agent** → **Tools**: Agent can call support tools when needed
- **Tools** → **Agent**: Tools return results back to agent for further processing

This creates a flexible ReAct (Reasoning + Acting) loop where the agent can:

1. Analyze customer messages
2. Decide whether to use tools
3. Process tool results
4. Provide final responses or continue the conversation

**Option A: Jupyter Notebook (Recommended)**

```bash
# Open the visualization notebook
jupyter notebook notebooks/graph_visualization.ipynb
```

**Option B: Python Script**

```bash
# Run the visualization script
python scripts/visualize_graph.py

# Or test the exact code you provided
python scripts/test_graph_viz.py
```

**Option C: Direct Code**

```python
from IPython.display import Image, display
from src.support_agent.agent import graph

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

**Required Dependencies for Visualization:**

```bash
pip install ipython pillow
pip install 'langgraph[visualization]'
```

### Debugging

Use LangGraph Studio to:

- Set breakpoints on nodes
- Step through execution
- Inspect state at any point
- Time-travel debug (rewind and replay)

### Code Style

We use:

- **Black** for formatting: `black src/`
- **Ruff** for linting: `ruff check src/`
- **Type hints** where possible

## Deployment

### Local Production

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.support_agent.agent:app
```

### Docker

```bash
# Build image
docker build -t support-bot .

# Run container
docker run -p 2024:2024 support-bot
```

## Common Patterns

### Adding Sentiment Analysis

```python
@tool
def analyze_sentiment(message: str) -> str:
    """Detect customer emotion."""
    # Use a sentiment model
    from transformers import pipeline
    sentiment = pipeline("sentiment-analysis")
    result = sentiment(message)[0]
    
    if result['label'] == 'NEGATIVE' and result['score'] > 0.8:
        return "customer_frustrated"
    return "neutral"
```

### Multi-step Workflows

```python
# For complex workflows, add conditional logic in agent.py
def should_follow_up(state: SupportState) -> str:
    last_msg = state["messages"][-1]
    if "thank" in last_msg.content.lower():
        return "end"
    return "continue"
```

## Resources

- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [Tool Calling Guide](https://python.langchain.com/docs/modules/agents/tools/)
- [Ollama Models](https://ollama.ai/library)
- [LangSmith Docs](https://docs.smith.langchain.com/)
- [Evaluation Guide](https://docs.smith.langchain.com/evaluation)

## Next Steps

- [ ] Add RAG with vector database (ChromaDB)
- [ ] Integrate with real order management system
- [ ] Add sentiment analysis for escalation
- [ ] Create web UI with React
- [ ] Add authentication and user management

## License

MIT License - Feel free to use and modify!

## Support

For issues or questions:

- Check the [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- Open an issue on GitHub
- Review example conversations in `scripts/test_bot.py`

---

**Quick Command Reference:**

```bash
# Run visually
langgraph dev

# Run in terminal
python scripts/cli.py

# Run tests
python scripts/test_bot.py

# Run evaluations
python -m src.support_agent.tests.eval_langsmith

# View examples
python scripts/api_client_example.py

# Test vector store
python scripts/init_vector_store.py

# Install deps
pip install -r requirements.txt

# Get Ollama model
ollama pull llama3.1:latest
```

Happy coding! 🚀
