# PROJECT OVERVIEW: Agentic Chatbot for Figure Q&A

## 1. Problem Statement

### Business Context
Chatbot thông minh có khả năng trả lời câu hỏi về nội dung của các figure (hình ảnh/biểu đồ) trong tài liệu. Chatbot cần:
- Hiểu ngữ cảnh câu hỏi và tự động quyết định cần thêm thông tin gì
- Lấy figure từ local storage khi cần phân tích hình ảnh
- Tìm kiếm thông tin bổ sung từ knowledge base (RAG)
- Trả lời chính xác và streaming response cho UX tốt

### Technical Challenges
- Multi-modal reasoning (text + image)
- Dynamic tool orchestration (conditional routing)
- Efficient context management với limited VRAM (16GB)
- Streaming response while hiding internal reasoning

---

## 2. System Overview

### High-Level Flow
```
User Query → Router Agent → [Get Figure / RAG / Direct Answer] → Context Aggregation → Final Answer
```

### Core Components
| Component | Responsibility |
|-----------|----------------|
| **Router Agent** | Phân tích query, quyết định cần tool nào |
| **Figure Tool (MCP)** | Lấy image base64 theo figure_id |
| **Semantic Search Tool (MCP)** | Semantic search trong ChromaDB, trả về context chunks |
| **Answer Generator** | Tổng hợp context và sinh câu trả lời |
| **Memory Manager** | Quản lý conversation history (PostgreSQL) |

---

## 3. Architecture

### 3.1 LangGraph Workflow Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    START                                 │
                    └─────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │              load_conversation_history                   │
                    │         (Load last 5 turns from PostgreSQL)             │
                    └─────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                  router_agent                            │
                    │    (Analyze query → Decide: get_figure/rag/answer)      │
                    └─────────────────────┬───────────────────────────────────┘
                                          │
                          ┌───────────────┼───────────────┐
                          │               │               │
                          ▼               ▼               ▼
              ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
              │  get_figure   │  │semantic_search│  │ direct_answer │
              │  (MCP Tool)   │  │  (MCP Tool)   │  │   (Skip)      │
              └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
                      │                   │                   │
                      └───────────────────┼───────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                aggregate_context                         │
                    │        (Merge figure + RAG results + history)           │
                    └─────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                generate_answer                           │
                    │      (Qwen3-VL-4B-Instruct → Stream final answer)       │
                    └─────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                save_to_memory                            │
                    │          (Save Q&A pair to PostgreSQL)                  │
                    └─────────────────────┬───────────────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                      END                                 │
                    └─────────────────────────────────────────────────────────┘
```

### 3.2 Node Descriptions

| Node | Type | Description |
|------|------|-------------|
| `load_conversation_history` | Function | Query PostgreSQL lấy 5 turns gần nhất theo `conversation_id` |
| `router_agent` | LLM Call | Gọi Qwen3-VL với structured output để quyết định actions cần thực hiện |
| `get_figure` | Tool (MCP) | Đọc figure từ local path, convert sang base64 bằng PIL |
| `semantic_search` | Tool (MCP) | Embed query bằng Qwen3-Embedding, search ChromaDB top-k |
| `aggregate_context` | Function | Merge tất cả context (figure, RAG, history) thành prompt |
| `generate_answer` | LLM Call | Gọi Qwen3-VL streaming, filter bỏ `<think>` tags |
| `save_to_memory` | Function | Insert conversation turn vào PostgreSQL |

### 3.3 Router Agent Output Schema

```json
{
  "actions": [
    {"type": "get_figure", "figure_id": "fig_001"},
    {"type": "semantic_search", "query": "explanation of concept X"}
  ],
  "ready_to_answer": false
}
```

Khi `ready_to_answer: true` → skip tools, đi thẳng `generate_answer`.

### 3.4 Conditional Routing Logic

```python
def route_decision(state: AgentState) -> list[str]:
    """Determine next nodes based on router output."""
    actions = state["router_output"]["actions"]
    next_nodes = []
    
    for action in actions:
        if action["type"] == "get_figure":
            next_nodes.append("get_figure")
        elif action["type"] == "semantic_search":
            next_nodes.append("semantic_search")
    
    if not next_nodes or state["router_output"]["ready_to_answer"]:
        next_nodes = ["aggregate_context"]
    
    return next_nodes
```

---

## 4. Tech Stack

### 4.1 Core Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **LLM Framework** | LangGraph | 0.2.x | Workflow orchestration |
| **LLM Serving** | vLLM | 0.6.x | High-performance inference |
| **Embedding Model** | Qwen3-Embedding-0.6B (vLLM API) | - | Text embedding cho RAG via vLLM |
| **Reasoning Model** | Qwen3-VL-4B-Instruct-FP8 | - | Multi-modal reasoning |
| **Vector DB** | ChromaDB | 0.5.x | Offline vector storage |
| **Database** | PostgreSQL | 16.x | Conversation history |
| **Backend** | FastAPI | 0.115.x | REST API + SSE streaming |
| **Frontend** | Streamlit | 1.40.x | Simple chat UI |
| **Tool Protocol** | MCP (Custom) | - | Standardized tool calling |

### 4.2 Python Dependencies

```
# Core
langchain>=0.3.0
langgraph>=0.2.0
fastapi>=0.115.0
uvicorn>=0.32.0
streamlit>=1.40.0

# LLM & Embedding
vllm>=0.6.0
openai>=1.50.0  # vLLM OpenAI-compatible client
transformers>=4.45.0

# Vector DB & RAG
chromadb>=0.5.0
pandas>=2.0.0
openpyxl>=3.1.0
httpx>=0.27.0

# Database
asyncpg>=0.30.0
sqlalchemy>=2.0.0

# Utilities
pillow>=10.0.0
pydantic>=2.0.0
python-multipart>=0.0.9
sse-starlette>=2.0.0
httpx>=0.27.0
loguru>=0.7.0
```

### 4.3 Hardware Requirements

| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA Quadro RTX 5000 (16GB VRAM) |
| RAM | 32GB recommended |
| Storage | SSD, ~50GB for models + data |

### 4.4 VRAM Allocation Strategy

```
Qwen3-VL-4B-Instruct-FP8 (70% GPU): ~11GB
Qwen3-Embedding-0.6B (10% GPU):     ~1.5GB
vLLM overhead + KV cache:           ~2GB
Buffer:                             ~1.5GB
─────────────────────────────────────────
Total:                              ~16GB ✓
```

---

## 5. Implementation Plan

### Phase 1: Infrastructure Setup (Day 1-2)

| Step | Task | Details |
|------|------|---------|
| 1.1 | Setup project structure | Tạo folder structure, virtual env |
| 1.2 | Setup PostgreSQL | Docker compose, create tables |
| 1.3 | Setup vLLM server | Load Qwen3-VL-4B-Thinking-FP8 |
| 1.4 | Setup Embedding server | Separate vLLM instance hoặc local loading |

### Phase 2: Core Components (Day 3-5)

| Step | Task | Details |
|------|------|---------|
| 2.1 | Implement MCP Figure Tool | PIL read → base64, tool schema |
| 2.2 | Build RAG pipeline | ChromaDB setup, indexing script, search function |
| 2.3 | Implement Memory Manager | PostgreSQL CRUD cho conversations |
| 2.4 | Build Router Agent | Prompt engineering, structured output parsing |

### Phase 3: LangGraph Workflow (Day 6-7)

| Step | Task | Details |
|------|------|---------|
| 3.1 | Define AgentState | TypedDict với tất cả state fields |
| 3.2 | Implement all nodes | Các function nodes cho workflow |
| 3.3 | Build conditional edges | Routing logic |
| 3.4 | Test workflow end-to-end | Unit tests + integration tests |

### Phase 4: API & Streaming (Day 8-9)

| Step | Task | Details |
|------|------|---------|
| 4.1 | FastAPI endpoints | `/chat` SSE endpoint, `/conversations` CRUD |
| 4.2 | Streaming handler | Filter thinking tokens, yield answer only |
| 4.3 | Error handling | Graceful degradation, retries |

### Phase 5: UI & Integration (Day 10-11)

| Step | Task | Details |
|------|------|---------|
| 5.1 | Streamlit chat UI | Chat interface với streaming |
| 5.2 | Conversation management | Select/create conversations |
| 5.3 | End-to-end testing | Full flow testing |

### Phase 6: Optimization & Deployment (Day 12-14)

| Step | Task | Details |
|------|------|---------|
| 6.1 | Performance tuning | Batch size, cache optimization |
| 6.2 | Docker containerization | Multi-stage builds |
| 6.3 | Documentation | API docs, deployment guide |

---

## 6. Project Structure

```
mps-chatbot/
├── docker-compose.yml              # PostgreSQL, vLLM services
├── pyproject.toml                  # Dependencies
├── .env.example                    # Environment variables template
├── README.md
├── PROJECT_OVERVIEW.md
│
├── scripts/
│   ├── index_documents.py          # Indexing documents vào ChromaDB
│   ├── setup_database.py           # Create PostgreSQL tables
│   └── start_vllm.sh               # vLLM server startup script
│
├── data/
│   ├── figures/                    # Local figure storage
│   │   └── {figure_id}.png
│   └── documents/                  # Documents for RAG indexing
│       └── [BCA]TERM_DATA.xlsx     # BCA terms dictionary
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py             # Pydantic settings management
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py              # Pydantic models (request/response)
│   │   └── state.py                # LangGraph AgentState definition
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── mcp_protocol.py         # MCP base classes and protocols
│   │   ├── figure_tool.py          # Get figure by ID → base64
│   │   ├── semantic_search_tool.py # Semantic search in vector DB
│   │   └── tool_registry.py        # Centralized tool management
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py           # PostgreSQL async connection
│   │   ├── models.py               # SQLAlchemy ORM models
│   │   ├── repository.py           # CRUD operations
│   │   └── volumes/                # Persistent storage
│   │       ├── chromadb/           # ChromaDB vector store
│   │       └── postgres_data/      # PostgreSQL data
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embedder.py             # vLLM Embedding API client
│   │   ├── vectorstore.py          # ChromaDB operations
│   │   └── retriever.py            # Semantic search logic
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py               # vLLM OpenAI-compatible client
│   │   ├── router.py               # Router agent logic
│   │   └── generator.py            # Answer generation với streaming
│   │
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── graph.py                # LangGraph workflow definition
│   │   ├── nodes.py                # All node implementations
│   │   └── edges.py                # Conditional edge functions
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app entry point
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── chat.py             # /chat SSE endpoint
│       │   └── conversations.py    # Conversation CRUD
│       └── middleware.py           # Error handling, logging
│
├── ui/
│   └── app.py                      # Streamlit chat application
│
└── tests/
    ├── __init__.py
    ├── test_tools.py
    ├── test_rag.py
    ├── test_workflow.py
    └── test_api.py
```

---

## 7. Key Implementation Details

### 7.1 Router Agent Prompt Template

```python
ROUTER_SYSTEM_PROMPT = """You are a routing agent that analyzes user queries about figures/documents.

Given a query and conversation history, decide what actions are needed:
1. "get_figure" - When query asks about visual content of a specific figure
2. "semantic_search" - When query needs additional context/explanation from knowledge base
3. Both - When query needs figure AND additional context
4. None - When you can answer directly from conversation history

Output JSON format:
{
  "reasoning": "brief explanation of your decision",
  "actions": [
    {"type": "get_figure", "figure_id": "xxx"},
    {"type": "semantic_search", "query": "reformulated search query"}
  ],
  "ready_to_answer": true/false
}

Current figure_id in context: {figure_id}
"""
```

---

## 8. Configuration

### 8.1 Environment Variables

```bash
# vLLM LLM Configuration
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct-FP8
VLLM_PORT=8000
VLLM_MAX_LEN=4096
VLLM_GPU_UTIL=0.70

# Embedding Configuration (vLLM API)
EMBEDDING_BASE_URL=http://localhost:8001/v1
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_PORT=8001
EMBEDDING_MAX_LEN=512
EMBEDDING_GPU_UTIL=0.10

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/chatbot

# ChromaDB
CHROMA_PERSIST_DIR=./src/database/volumes/chromadb
CHROMA_COLLECTION_NAME=bca_terms

# Figure Storage
FIGURES_DIR=./data/figures

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
```

### 8.2 vLLM Startup Command

```bash
# Use start_vllm.sh to start both servers (reads from .env)
./scripts/start_vllm.sh

# Or manually start LLM server (port 8000, 70% GPU)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-4B-Instruct-FP8 \
    --port 8000 --max-model-len 4096 \
    --gpu-memory-utilization 0.70 --trust-remote-code

# And Embedding server (port 8001, 10% GPU)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Embedding-0.6B \
    --port 8001 --max-model-len 512 \
    --gpu-memory-utilization 0.10 --trust-remote-code
```

---

## 9. Success Criteria

| Metric | Target |
|--------|--------|
| Response latency (first token) | < 2s |
| Streaming throughput | > 20 tokens/s |
| RAG retrieval accuracy | > 80% relevance |
| Memory usage | < 16GB VRAM |
| Uptime | > 99% |

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| VRAM overflow | High | FP8 quantization, careful batch sizing |
| Slow RAG retrieval | Medium | Pre-compute embeddings, optimize ChromaDB |
| Router misclassification | Medium | Few-shot examples trong prompt, fallback logic |
| PostgreSQL connection issues | Low | Connection pooling, retry logic |

---

## 11. Future Enhancements

1. **Multi-turn figure comparison** - So sánh nhiều figures trong 1 conversation
2. **Proactive suggestions** - Gợi ý câu hỏi liên quan
3. **Export conversations** - Xuất conversation ra PDF/Markdown
4. **Admin dashboard** - Monitor usage, analytics
5. **Fine-tuning** - Fine-tune router trên domain-specific data
