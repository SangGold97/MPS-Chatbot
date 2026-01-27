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
| **RAG System** | Semantic search trong ChromaDB |
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
              │  get_figure   │  │   rag_search  │  │ direct_answer │
              │  (MCP Tool)   │  │  (ChromaDB)   │  │   (Skip)      │
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
                    │     (Qwen3-VL-4B-Thinking → Stream final answer)        │
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
| `rag_search` | Function | Embed query bằng Qwen3-Embedding, search ChromaDB top-k |
| `aggregate_context` | Function | Merge tất cả context (figure, RAG, history) thành prompt |
| `generate_answer` | LLM Call | Gọi Qwen3-VL streaming, filter bỏ `<think>` tags |
| `save_to_memory` | Function | Insert conversation turn vào PostgreSQL |

### 3.3 Router Agent Output Schema

```json
{
  "actions": [
    {"type": "get_figure", "figure_id": "fig_001"},
    {"type": "rag", "query": "explanation of concept X"}
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
        elif action["type"] == "rag":
            next_nodes.append("rag_search")
    
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
| **Embedding Model** | Qwen3-Embedding-0.6B | - | Text embedding cho RAG |
| **Reasoning Model** | Qwen3-VL-4B-Thinking-FP8 | - | Multi-modal reasoning |
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
sentence-transformers>=3.0.0

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
Qwen3-VL-4B-Thinking-FP8:           ~8GB
Qwen3-Embedding-0.6B:               ~2GB
vLLM overhead + KV cache:           ~4GB
Buffer:                             ~2GB
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
│   ├── documents/                  # Documents for RAG indexing
│   └── chroma_db/                  # ChromaDB persistent storage
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
│   │   ├── mcp_server.py           # MCP tool server implementation
│   │   ├── figure_tool.py          # Get figure by ID → base64
│   │   └── rag_tool.py             # RAG search implementation
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py           # PostgreSQL async connection
│   │   ├── models.py               # SQLAlchemy ORM models
│   │   └── repository.py           # CRUD operations
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embedder.py             # Qwen3-Embedding wrapper
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
2. "rag" - When query needs additional context/explanation from knowledge base
3. Both - When query needs figure AND additional context
4. None - When you can answer directly from conversation history

Output JSON format:
{
  "reasoning": "brief explanation of your decision",
  "actions": [
    {"type": "get_figure", "figure_id": "xxx"},
    {"type": "rag", "query": "reformulated search query"}
  ],
  "ready_to_answer": true/false
}

Current figure_id in context: {figure_id}
"""
```

### 7.2 Streaming Response Filter

```python
async def filter_thinking_tokens(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Filter out <think>...</think> blocks from streaming response."""
    buffer = ""
    in_thinking = False
    
    async for chunk in stream:
        buffer += chunk
        
        while buffer:
            if in_thinking:
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    buffer = buffer[end_idx + 8:]
                    in_thinking = False
                else:
                    break
            else:
                start_idx = buffer.find("<think>")
                if start_idx != -1:
                    yield buffer[:start_idx]
                    buffer = buffer[start_idx + 7:]
                    in_thinking = True
                else:
                    # Yield tất cả trừ potential partial tag
                    safe_len = len(buffer) - 7
                    if safe_len > 0:
                        yield buffer[:safe_len]
                        buffer = buffer[safe_len:]
                    break
    
    if buffer and not in_thinking:
        yield buffer
```

### 7.3 Database Schema

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- 'user' | 'assistant'
    content TEXT NOT NULL,
    figure_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at DESC);
```

---

## 8. Configuration

### 8.1 Environment Variables

```bash
# vLLM Configuration
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=Qwen/Qwen3-VL-4B-Thinking-FP8

# Embedding Configuration  
EMBEDDING_MODEL_PATH=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DEVICE=cuda

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/chatbot

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION_NAME=documents

# Figure Storage
FIGURES_DIR=./data/figures

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
```

### 8.2 vLLM Startup Command

```bash
# Start vLLM with Qwen3-VL-4B-Thinking-FP8
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-4B-Thinking-FP8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --trust-remote-code
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
