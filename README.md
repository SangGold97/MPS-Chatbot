# MPS Chatbot — Agentic Figure Q&A Chatbot

An agentic chatbot built with **LangGraph** that answers questions about figures (charts/images) and biology terminology. The system dynamically decides which tools to invoke — retrieving figure images or searching a knowledge base — before generating a streaming answer via a multimodal VLM.

---

## Features

- **Multimodal Q&A**: Analyze figure images (PNG/JPG) with a vision-language model
- **RAG pipeline**: Semantic search over a ChromaDB knowledge base (BCA biology terms)
- **Agentic routing**: Router LLM decides at runtime which tools to call in parallel
- **Conversation memory**: Per-session history persisted in PostgreSQL (last 5 turns)
- **Streaming output**: Token-by-token streaming response via `stream_generate()`
- **MCP-style tools**: Standardized tool protocol with typed schemas (`MCPTool` ABC)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Workflow orchestration | LangGraph |
| LLM & VLM | Qwen3-VL-4B-Instruct-FP8 (via vLLM) |
| Embedding model | Qwen3-Embedding-0.6B (via vLLM) |
| Vector database | ChromaDB (persistent, exact FLAT index) |
| Conversation memory | PostgreSQL 16 (asyncpg) |
| LLM client | LangChain-OpenAI → vLLM OpenAI-compatible API |
| Image processing | Pillow (PIL) |

---

## Directory Structure

```
MPS-Chatbot/
├── docker-compose.yml          # PostgreSQL service (port 5433)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (see Configuration)
│
├── scripts/
│   ├── index_documents.py      # Index XLSX documents into ChromaDB
│   └── start_vllm.sh           # Start both vLLM servers (LLM + Embedding)
│
├── data/
│   ├── figures/                # Figure images referenced by figure_id
│   └── documents/
│       └── [BCA]TERM_DATA.xlsx # BCA terminology dataset for RAG
│
├── src/
│   ├── config/settings.py      # Pydantic settings loaded from .env
│   ├── models/
│   │   ├── schemas.py          # RouterOutput, RouterAction, ChatRequest/Response
│   │   └── state.py            # AgentState TypedDict (LangGraph state)
│   ├── tools/
│   │   ├── mcp_protocol.py     # MCPTool ABC + MCPToolOutput base class
│   │   ├── figure_tool.py      # FigureTool: load image → base64 via PIL
│   │   ├── semantic_search_tool.py  # SemanticSearchTool: embed → ChromaDB query
│   │   └── tool_registry.py    # ToolRegistry singleton managing both tools
│   ├── database/
│   │   └── postgres_manager.py # Async PostgreSQL CRUD (asyncpg connection pool)
│   ├── rag/
│   │   └── embedder.py         # EmbeddingClient: calls vLLM /embeddings API
│   ├── llm/
│   │   ├── client.py           # LLMClient: async OpenAI-compat wrapper for vLLM
│   │   ├── router.py           # RouterAgent: structured output via LangChain
│   │   └── generator.py        # AnswerGenerator: generate() + stream_generate()
│   ├── workflow/
│   │   ├── graph.py            # LangGraph StateGraph + compiled singleton
│   │   ├── nodes.py            # All async node functions
│   │   └── edges.py            # route_decision() conditional edge logic
│   └── api/                    # [Planned] FastAPI REST + SSE endpoints
│
├── ui/                         # [Planned] Streamlit chat UI
│
└── tests/
    ├── test_indexing.py        # ChromaDB indexing tests
    ├── test_llm.py             # LLM client tests
    ├── test_postgres.py        # PostgreSQL manager tests
    ├── test_tools.py           # MCP tool tests
    └── test_workflow.py        # Full pipeline integration test (terminal demo)
```

---

## Workflow

The chatbot is orchestrated as a **LangGraph** state machine. Each user query flows through the following nodes:

```mermaid
flowchart TD
    A([START]) --> B["load_conversation_history\n(PostgreSQL: last 5 turns)"]
    B --> C["router_agent\n(Qwen3-VL structured output\n→ RouterOutput)"]

    C -->|"actions: [get_figure]"| D["get_figure\n(PIL → resize → base64)"]
    C -->|"actions: [semantic_search]"| E["semantic_search\n(Qwen3-Embedding → ChromaDB)"]
    C -->|"ready_to_answer: true\nor no actions"| F

    D --> F["aggregate_context\n(merge figure + RAG chunks\n+ history → LLM messages)"]
    E --> F

    F --> G["generate_answer\n(Qwen3-VL via vLLM)"]
    G --> H["save_to_memory\n(INSERT turn → PostgreSQL)"]
    H --> I([END])
```

### Node Descriptions

| Node | Description |
|------|-------------|
| `load_conversation_history` | Fetches the last 5 turns from PostgreSQL as `[user, assistant]` message pairs |
| `router_agent` | Calls Qwen3-VL with `with_structured_output(RouterOutput)` to decide which tools to invoke |
| `get_figure` | Resolves `figure_id` to a local file in `data/figures/`, resizes with PIL, returns base64 |
| `semantic_search` | Encodes the (optionally rewritten) query via Qwen3-Embedding, queries ChromaDB top-k |
| `aggregate_context` | Builds the final message list: system prompt + RAG context + history + user turn (with image if present) |
| `generate_answer` | Calls vLLM LLM for the final answer (non-streaming in graph; streaming exposed via `AnswerGenerator.stream_generate()`) |
| `save_to_memory` | Inserts the Q&A turn into PostgreSQL |

### Router Output Schema

```json
{
  "reasoning": "User is asking about a figure, need to retrieve it.",
  "actions": [
    {"type": "get_figure", "figure_id": "bar3"},
    {"type": "semantic_search", "query": "BCA related concepts"}
  ],
  "ready_to_answer": false
}
```

When `ready_to_answer: true`, the router skips all tools and routes directly to `aggregate_context`.  
When multiple actions are present, `get_figure` and `semantic_search` execute **in parallel** (LangGraph fan-out), then fan-in to `aggregate_context`.

---

## Hardware Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | NVIDIA GPU with ≥16 GB VRAM (tested on Quadro RTX 5000) |
| RAM | 32 GB recommended |
| Storage | ~50 GB SSD (models + data) |

**VRAM allocation:**

```
Qwen3-VL-4B-Instruct-FP8  (70% GPU)  ≈ 11 GB
Qwen3-Embedding-0.6B       (10% GPU)  ≈  1.5 GB
vLLM overhead + KV cache              ≈  2 GB
Buffer                                ≈  1.5 GB
─────────────────────────────────────────────
Total                                 ≈ 16 GB
```

---

## Installation & Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd MPS-Chatbot

conda activate mps   # or use your preferred virtualenv
pip install -r requirements.txt

# Additional packages needed for indexing only
pip install pandas openpyxl
```

### 2. Configure environment

Create a `.env` file in the project root:

```bash
# vLLM LLM server
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct-FP8
VLLM_PORT=8000
VLLM_MAX_LEN=4096
VLLM_GPU_UTIL=0.70

# vLLM Embedding server
EMBEDDING_BASE_URL=http://localhost:8001/v1
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_PORT=8001
EMBEDDING_MAX_LEN=512
EMBEDDING_GPU_UTIL=0.10

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/chatbot

# ChromaDB
CHROMA_PERSIST_DIR=./src/database/volumes/chromadb
CHROMA_COLLECTION_NAME=bca_terms

# Figures
FIGURES_DIR=./data/figures
```

### 3. Start PostgreSQL

```bash
docker compose up -d
```

### 4. Start vLLM servers

```bash
bash scripts/start_vllm.sh
```

This launches two vLLM processes:
- **Port 8000** — `Qwen3-VL-4B-Instruct-FP8` (LLM + VLM, 70% GPU)
- **Port 8001** — `Qwen3-Embedding-0.6B` (embeddings, 10% GPU)

### 5. Index documents into ChromaDB

Place your `.xlsx` file (with `Term` and `Description` columns) in `data/documents/`, then:

```bash
python scripts/index_documents.py
```

This reads `[BCA]TERM_DATA.xlsx`, generates embeddings via vLLM, and stores them in ChromaDB.

### 6. Add figures

Copy figure images into `data/figures/`. Files are referenced by filename stem (e.g., `bar3.png` → `figure_id="bar3"`).

---

## Running the Chatbot

### Terminal demo (full pipeline with streaming)

The integration test in `tests/test_workflow.py` runs the complete workflow step-by-step and streams the answer to the terminal:

```bash
conda activate mps
python tests/test_workflow.py
```

**Example output:**

```
============================================================
2025-xx-xx | INFO | Step 1: load_conversation_history
2025-xx-xx | INFO |   → history: 0 messages
------------------------------------------------------------
2025-xx-xx | INFO | Step 2: router_agent
2025-xx-xx | INFO |   → reasoning: User asks about a figure, retrieving it.
2025-xx-xx | INFO |   → actions: [{'type': 'get_figure', 'figure_id': 'bar3.png'}]
2025-xx-xx | INFO |   → ready_to_answer: False
------------------------------------------------------------
2025-xx-xx | INFO | Step 3: route_decision → ['get_figure']
...
2025-xx-xx | INFO | Step 6: generate_answer (streaming)

The bar chart shows ...   <streamed tokens appear here in real time>
============================================================
```

To test a different query, edit `main()` in `test_workflow.py`:

```python
await run_step_by_step(
    query="BCA là gì?",
    figure_id=None,        # No figure → triggers semantic_search only
)
```

### Running individual tests

```bash
python tests/test_llm.py        # Test LLM client
python tests/test_tools.py      # Test MCP tools
python tests/test_postgres.py   # Test PostgreSQL manager
python tests/test_indexing.py   # Test ChromaDB indexing
```

---

## Configuration Reference

All settings are managed by `src/config/settings.py` (Pydantic `BaseSettings`). Values are loaded from `.env` with the defaults shown above. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL_NAME` | `Qwen/Qwen3-VL-4B-Instruct-FP8` | VLM model served by vLLM |
| `EMBEDDING_MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model for RAG |
| `CHROMA_COLLECTION_NAME` | `bca_terms` | ChromaDB collection name |
| `FIGURES_DIR` | `./data/figures` | Directory containing figure images |
| `DATABASE_URL` | — | asyncpg-compatible PostgreSQL URL |

---

## Roadmap

- [ ] FastAPI REST + SSE streaming endpoint (`src/api/`)
- [ ] Streamlit chat UI (`ui/app.py`)
- [ ] Multi-figure comparison in a single conversation turn
- [ ] Conversation export (PDF / Markdown)
