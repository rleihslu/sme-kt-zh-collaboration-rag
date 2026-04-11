# Deployment Analysis

> Living document — updated as we progress toward a company deployment.

---

## Stack Overview

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15.5, React 18, TypeScript, Tailwind CSS, Radix UI |
| Backend | FastAPI (uvicorn), Python 3.11+ |
| LLM | OpenAI (default) or Ollama (local) |
| Vector Store | ChromaDB (text + image embeddings, in-process) |
| Storage | Flat JSON files (conversations, messages, reactions, users) |
| Auth | Cookie-based JWT + optional shared passcode |
| i18n | en, de, fr, it (via i18next) |

---

## How the App Is Structured

The frontend is a **Next.js app** built to a static `dist/` folder. In production, the FastAPI backend serves the frontend directly as static files — so the whole application runs as a single process on port 8080.

**Key source locations:**
- `frontend/src/pages/` — Next.js pages (`index.tsx`, `passcode.tsx`)
- `frontend/src/components/` — UI components (sidebar, message list, send bar, header)
- `frontend/src/services/` — API clients (conversations, messages, passcode)
- `frontend/src/config.ts` — Branding (currently "PrimePack AG", agent "Prime")
- `backend/src/.../auth/` — Auth providers (`SessionCookieProvider`, `PasscodeProvider`)
- `backend/src/.../main.py` — FastAPI app entry point, RAG pipeline wiring

---

## Phase 1 — Prototype Deployment

**Goal:** Get the app running for a small group of internal users with minimal infrastructure changes. Validate the RAG use case before investing in production infrastructure.

### What's already there

- **Per-user conversation isolation** — conversations are scoped to a `user_id` server-side; users cannot access each other's data
- **Anonymous session auth** — JWT auto-generated on first visit, stored as a cookie (365-day lifetime); no login UI required
- **Optional passcode gate** — a single shared password can restrict access (`PASSCODE` env var); good enough for a controlled pilot
- **Streaming responses** — SSE-based streaming is already implemented
- **Conversation history** — persisted server-side in JSON files; survives browser restarts as long as the user returns from the same browser

### Key limitation in Phase 1

Conversation history is tied to the **browser cookie**, not to a user identity. If a user switches to a different browser or device, they get a new anonymous session and lose access to their history. For a prototype with a small, informed group of users this is acceptable — but it must be resolved before Phase 2.

### Deployment approach — direct install, no Docker needed

In a VMware environment, running Docker inside a VM adds an unnecessary layer for a prototype. **Direct installation on the VM is recommended for Phase 1.** The app is set up as a Python virtualenv with a systemd service keeping FastAPI running, and nginx in front for HTTPS. This is simpler, has fewer moving parts, and is fully sufficient for a small pilot.

Docker becomes more valuable in Phase 2 when multiple services (RAG app, Open WebUI, AnythingLLM) need to be managed and updated consistently.

### What needs to be done for Phase 1

| Task | Details |
|------|---------|
| App VM setup | Python venv, install dependencies, build frontend static assets (`npm run build`) |
| systemd service | Run FastAPI as a managed service — auto-restarts on failure or reboot |
| Fix secret key | Override `SECRET_KEY` (currently defaults to `"1234567890"`) in the environment |
| Set `ENV=production` | Ensures cookies are set with `Secure` + `HttpOnly` flags |
| nginx + HTTPS | TLS termination; reverse proxy to FastAPI on port 8080 |
| Enable passcode | Set `PASSCODE` env var to gate access during the pilot |
| Persistent storage | Ensure `DB_DIR` points to a volume that survives VM snapshots and rebuilds |
| Set up inference VM | Spin up a second VM (CPU-only) running Ollama with a small local model |

### Local LLM for Phase 1 — CPU-only VM

To avoid sending data to the cloud even during the prototype, a second CPU-only VM runs **Ollama**, which exposes an OpenAI-compatible API. The RAG backend just needs `SERVER_URL` pointed at this VM — identical to how it will work in Phase 2 with vLLM on the Spark.

> Ollama is preferred over vLLM for CPU-only inference — vLLM is GPU-optimised and adds unnecessary complexity on CPU hardware.

**Recommended model for CPU testing:**

| Model | Size (4-bit quant) | Notes |
|-------|-------------------|-------|
| **Gemma 4 E2B** | ~3 GB model weights | **Best choice.** 128K context, 140+ languages, manageable on CPU. Staying in the Gemma 4 family means prompts and behavior validated in Phase 1 carry directly over to Phase 2. |
| **Gemma 4 E4B** | ~5 GB model weights | Better quality, same capabilities; noticeably slower on CPU — worth it if the VM has 16 GB+ RAM and response speed is acceptable |
| **Llama 3.2 3B** | ~2 GB model weights | Fallback; fastest on CPU, well-supported in Ollama, but smaller context and weaker multilingual support |

**Recommended VM specs for the inference VM:**

| Resource | Option A | Option B | Notes |
|----------|---------|---------|-------|
| vCPU | 8 | 8 | More cores = faster token generation; llama.cpp (used by Ollama) parallelises well across cores |
| RAM | 16 GB | 32 GB | See note below |
| Disk | 20 GB | 20 GB | Model files + OS |
| GPU | None | None | CPU-only for Phase 1 |

**16 GB RAM (Option A):** Comfortable for Gemma 4 E2B at 4-bit (~3 GB model weights). Leaves sufficient headroom for OS, Ollama overhead, and the KV cache for typical conversation lengths. Recommended starting point.

**32 GB RAM (Option B):** Opens up meaningfully better options:
- Run **Gemma 4 E4B** (4-bit, ~5 GB) with plenty of headroom — noticeably stronger quality than E2B
- Larger KV cache → handles longer conversations and bigger retrieved contexts without degradation
- If the App VM and Inference VM are consolidated onto a single larger VM to save resources, 32 GB gives room for both workloads
- Generally future-proof for Phase 1 experimentation

> Expect 3–8 tokens/sec on a modern CPU for E2B, slightly less for E4B — noticeably slower than a GPU, but usable for a prototype. Set user expectations accordingly.

### Where chunking and retrieval run — and how it's controlled

The RAG pipeline has two distinct phases, and it matters where each runs:

**1. Ingestion / chunking** (offline, run once or periodically):
Splits documents into chunks, generates embeddings, and writes them to ChromaDB. In Phase 1 this is triggered manually as a Python script on the **App VM**. It's a batch process — not part of the live request path.

**2. Retrieval** (online, per query):
When a user asks a question, the backend performs hybrid search (BM25 + vector similarity) against ChromaDB and assembles a context window before calling the LLM.

**How the split is controlled today:** A single environment variable — `SERVER_URL` — points at the LLM endpoint. Everything else (retrieval, ChromaDB, conversation management, frontend) stays on the App VM. Switching inference backends is a one-line config change.

#### Phase 1 — retrieval on App VM (acceptable)

For a prototype with a small knowledge base (hundreds to low thousands of chunks), running retrieval in-process on the App VM is fine. The CPU handles it without issue at that scale.

```
Query arrives
  → BM25 (keyword search, App VM CPU)
  → query embedding (embedding model, App VM CPU)  ← slow on CPU
  → vector similarity search (ChromaDB, in-process, App VM)
  → context assembled
  → HTTP POST to SERVER_URL → Ollama on Inference VM
  ← streamed response
```

The weak point here is the **query embedding step** — running the embedding model on CPU adds latency to every query. Acceptable for a prototype, but noticeable.

#### Phase 2 — retrieval moves to the Spark

Retrieval can be genuinely performance-hungry, especially as the knowledge base grows:

- **Query embedding** — converting the incoming query to a vector requires running a transformer model (e.g. `nomic-embed-text` or `bge-m3`). On GPU this takes ~10–50 ms; on CPU it can be 500 ms–2 seconds per query.
- **Vector similarity search** — comparing the query vector against tens or hundreds of thousands of stored embeddings is a matrix operation. ChromaDB can use GPU acceleration for this; on CPU it grows linearly with collection size.
- **Ingestion / re-indexing** — when new documents are added, embedding all chunks is heavily GPU-accelerated. A batch that takes hours on CPU takes minutes on GPU.

**In Phase 2, the Spark handles all compute-heavy AI work**, and the App VM becomes a thin orchestration layer:

| Component | Phase 1 | Phase 2 |
|-----------|---------|---------|
| BM25 search | App VM (CPU) | App VM (CPU) — lightweight, stays here |
| Query embedding | App VM (CPU) — slow | Spark (GPU) — fast |
| Vector similarity search | App VM (ChromaDB in-process) | Spark (ChromaDB standalone service) |
| Ingestion & chunk embedding | App VM (CPU) | Spark (GPU batch jobs) |
| LLM generation | Inference VM (Ollama, CPU) | Spark (vLLM, GPU) |

```
Phase 2 query flow:

Query arrives at App VM
  → BM25 search (App VM, lightweight)
  → query text sent to Spark embedding service
  ← embedding vector returned
  → vector similarity search (ChromaDB on Spark)
  ← top-k chunks returned
  → context assembled on App VM
  → HTTP POST to vLLM on Spark
  ← streamed response to user
```

The App VM's job is orchestration and serving — auth, conversation history, context assembly, and routing. All the heavy compute lives on the Spark, which is purpose-built for it.

### Phase 1 architecture (two VMs, direct install)

```
Users (browser)
     │  HTTPS
     ▼
[App VM — direct install]
  nginx (TLS termination, port 443)
     │
  FastAPI / uvicorn (port 8080, systemd service)
  ├── serves frontend (static Next.js build)
  ├── RAG retrieval (BM25 + vector search)
  ├── ChromaDB (in-process, vector store)
  ├── Conversation storage (JSON files on persistent volume)
  └── Ingestion pipeline (run manually as needed)
     │
     │  internal network (OpenAI-compatible API)
     ▼
[Inference VM — CPU only, direct install]
  Ollama (systemd service)
  └── Gemma 4 E2B (or E4B / Llama 3.2 3B)
```

---

## Phase 2 — Production Deployment

**Goal:** Stable, secure, multi-user deployment for day-to-day company use. Full identity integration, proper data persistence, and fully on-premises LLM and retrieval on dedicated hardware. No data leaves the company network.

### Planned hardware

**NVIDIA Spark** (GB10 Grace Blackwell, ~128 GB unified memory) — dedicated AI server running:
- **LLM inference** via vLLM (see model candidates below)
- **Embedding model** for vector search
- **ChromaDB** vector store (as a standalone service)
- **Ingestion & chunking pipeline** (GPU-accelerated, run as batch jobs)

The app server (virtualized) and AI server (Spark) are on the same internal network and communicate over HTTP. They could initially run on the same machine and be separated later without architectural changes.

### LLM serving — vLLM

vLLM is the chosen inference server for Phase 2. Key reasons:
- Handles concurrent requests and request queueing natively (continuous batching)
- Exposes an OpenAI-compatible API — the RAG backend only needs its `SERVER_URL` pointed at the Spark instead of `api.openai.com`; no other code changes required
- Significantly more efficient than Ollama under multi-user load
- Supports PagedAttention for high-throughput inference

### LLM model candidates

| Model | Params (active) | Context | Notes |
|-------|----------------|---------|-------|
| **Gemma 4 26B A4B MoE** ✓ | 3.8B active / 25.2B total | 256K | **Recommended.** Mixture-of-Experts: only 3.8B params active per token → fast inference and low latency under concurrent load, while the full model quality is that of a 26B model. 256K context is ideal for RAG with long retrieved passages. 140+ languages. |
| **Gemma 4 31B Dense** | 30.7B | 256K | Strongest Gemma 4 variant; 256K context; higher quality ceiling but slower than MoE — worth considering if answer quality is more important than throughput |
| **Llama 3.3 70B** | 70B | 128K | Fully open weights; well-supported in vLLM; strong general-purpose performance; smaller context window than Gemma 4 |
| **Qwen 2.5 72B** (Alibaba) | 72B | 128K | Excellent multilingual and reasoning benchmarks; ⚠ Chinese-developed — consider data governance and trust policy before adopting in a company setting |

> **Why Gemma 4 26B A4B MoE for RAG?** The 256K context window allows including many retrieved chunks without truncation. The MoE architecture means inference cost scales with active parameters (3.8B), not total parameters — giving near-70B quality at near-7B speed. With max ~5 concurrent users and the Spark's 128 GB unified memory, this model will run comfortably with headroom to spare.

### Additional interfaces — shared AI backend

A key architectural principle: **vLLM is the single LLM backend, and multiple frontends point to it.** This avoids running duplicate model instances and lets users choose the interface that fits their workflow.

Planned interfaces in addition to the RAG app:

| Interface | Use case |
|-----------|---------|
| **This RAG app** | Knowledge base Q&A with source attribution and document retrieval |
| **Open WebUI** | General-purpose chat directly against the LLM; supports model selection, system prompts, conversation history |
| **AnythingLLM** | Alternative with built-in RAG management UI, document upload, and workspace isolation — could complement this app |

All three talk to the same vLLM endpoint. Users get two or more entry points depending on what they need, all powered by the same model running on the Spark.

### Authentication — Local Active Directory

The deployment is fully on-premises (virtualized app server + Spark). Auth will use the company's **local Active Directory via AD FS** (Active Directory Federation Services), which exposes a standard OIDC/OAuth2 endpoint that the FastAPI backend can integrate with directly.

Microsoft Entra ID (Azure AD) remains a secondary option if the company moves toward cloud-managed identities in the future — the integration point in the code is identical.

**Integration:** The FastAPI backend's `SessionCookieProvider` is replaced with an OIDC middleware that validates tokens issued by AD FS. Open WebUI and AnythingLLM also support OIDC natively, so all three interfaces can share the same AD FS integration.

**Impact on conversation history:** Once users authenticate with their AD identity, the `user_id` in the system becomes their stable corporate account — conversation history persists across all devices and browsers automatically.

### Phase 2 architecture

```
Users (browser)
     │  HTTPS
     ▼
[Reverse proxy — nginx/Traefik]
     ├──/          → RAG App (this repo)
     ├──/chat      → Open WebUI
     └──/docs      → AnythingLLM  (optional)
          │
          ▼
[App Server — VM]
  ├── RAG FastAPI backend
  │     ├── serves frontend (static Next.js build)
  │     ├── RAG retrieval & conversation API
  │     ├── OIDC auth → Active Directory / Entra ID
  │     └── Postgres (conversations, messages, reactions, users)
  ├── Open WebUI container
  └── AnythingLLM container (optional)
          │
          │  internal network (OpenAI-compatible API)
          ▼
[NVIDIA Spark — AI Server]
  ├── vLLM  (LLM inference, concurrent request handling)
  │     └── model: Llama 3.3 70B / Gemma 4 27B / TBD
  ├── Embedding model service
  ├── ChromaDB (vector store — standalone service)
  └── Ingestion & chunking pipeline (batch jobs)
```

### What changes from Phase 1

| Concern | Phase 1 | Phase 2 |
|---------|---------|---------|
| LLM serving | Ollama on CPU VM (on-premises) | vLLM on NVIDIA Spark (on-premises) |
| LLM model | Gemma 4 E2B (CPU, 4-bit) | Gemma 4 26B A4B MoE (GPU) — TBD after benchmarks |
| Auth | Anonymous cookie + passcode | OIDC via Active Directory / Entra ID |
| Conversation history | Tied to browser cookie | Tied to AD identity — persists across devices |
| Storage | Flat JSON files | Postgres |
| Vector store | In-process ChromaDB | ChromaDB as standalone service on Spark |
| Ingestion | Manual / notebook | Scheduled batch jobs on Spark |
| User interfaces | RAG app only | RAG app + Open WebUI + optionally AnythingLLM |
| Branding | "PrimePack AG" / "Prime" | Customized to target company |

### Code changes needed to support remote retrieval (Phase 2)

Currently the codebase has no mechanism to separate retrieval from the App VM — two things are hardwired:

**1. ChromaDB is always local**
`conversational-toolkit/src/conversational_toolkit/vectorstores/chromadb.py` always instantiates a `chromadb.PersistentClient(path=db_path)` — a local, in-process connection. ChromaDB also supports an HTTP client mode (`chromadb.HttpClient(host, port)`) for connecting to a standalone server, but there is no config to switch between them. A `CHROMA_HOST` / `CHROMA_PORT` env var needs to be added, and the client instantiation updated to use `HttpClient` when those are set.

**2. The embedding model is hardcoded to OpenAI**
`backend/src/.../feature0_baseline_rag.py` hardcodes `EMBEDDING_MODEL = "text-embedding-3-small"`, and the `OpenAIEmbeddings` class calls `AsyncOpenAI()` with no `base_url` — so it always calls `api.openai.com`. For Phase 2, the embedding service on the Spark (served by vLLM or a dedicated service like Infinity) exposes the same OpenAI-compatible `/v1/embeddings` endpoint. The only change needed is passing a `base_url` to `AsyncOpenAI()` and making the model name configurable via env var.

**Summary of changes required:**

| File | Change |
|------|--------|
| `conversational-toolkit/.../vectorstores/chromadb.py` | Add `CHROMA_HOST`/`CHROMA_PORT` env vars; use `chromadb.HttpClient` when set, `PersistentClient` otherwise |
| `backend/src/.../feature0_baseline_rag.py` | Make `EMBEDDING_MODEL` an env var (default: `text-embedding-3-small` for Phase 1 compatibility) |
| `backend/src/.../embeddings/openai.py` | Accept and pass `base_url` to `AsyncOpenAI()`; add `EMBEDDING_URL` env var |
| `backend/src/.../main.py` | Pass `EMBEDDING_URL` through to `OpenAIEmbeddings` constructor |

**Image embedding (`Qwen3VLEmbeddings`)** also needs to move to the Spark in Phase 2, but requires different treatment. The default model is `Qwen/Qwen3-VL-Embedding-2B` — only 2B parameters, so it fits comfortably alongside the main LLM (see memory budget below). However, it is not OpenAI-API-compatible and cannot be pointed at vLLM. It needs to be wrapped in a small custom FastAPI microservice on the Spark, which the App VM calls via an `IMAGE_EMBEDDING_URL` env var. An additional table row for `main.py` would wire this up.

**Spark memory budget (Phase 2):**

| Service | Approx. memory |
|---------|---------------|
| Gemma 4 26B MoE — vLLM (bf16) | ~50 GB |
| Qwen3-VL-Embedding-2B (bf16) | ~4 GB |
| Text embedding model (e.g. bge-m3) | ~2 GB |
| ChromaDB indexes + OS overhead | ~10–15 GB |
| **Total** | **~66–71 GB** — ~60 GB headroom on the 128 GB Spark |

These are all well-contained changes. In Phase 1 none of them are needed — everything runs locally as-is. They become relevant only when preparing for Phase 2 and moving retrieval to the Spark.

### Missing feature flag for image embedding

The codebase uses a `feature0_`, `feature1_`, `feature3_`, `feature4_` file naming convention suggesting features should be gated individually. However, `main.py` imports and instantiates `Qwen3VLEmbeddings` unconditionally — there is no env var or flag to disable the image embedding pipeline at startup.

In practice this means: on every server start, the full `Qwen/Qwen3-VL-Embedding-2B` model is loaded into memory via HuggingFace `transformers`, even if the knowledge base contains no images. On a CPU-only App VM with limited RAM this will cause a very slow startup or an out-of-memory failure.

**Recommended fix before Phase 1 deployment:** add an `ENABLE_IMAGE_RETRIEVAL` env var (default: `false`) that gates the `Qwen3VLEmbeddings` instantiation and the `image_retriever` setup in `build_server()`. This restores the feature-flag pattern used elsewhere in the codebase and makes the prototype deployment safe on CPU hardware.

### Things to watch out for

**vLLM compatibility with the NVIDIA Spark (GB10 Grace Blackwell)**
The Spark uses an ARM-based Grace CPU paired with a Blackwell GPU. vLLM primarily targets x86 + NVIDIA GPU; ARM/Grace support is still maturing. Verify compatibility before committing fully to vLLM on this hardware. If there are issues, llama.cpp server and HuggingFace TGI (Text Generation Inference) are drop-in alternatives — both also expose an OpenAI-compatible API.

**Ingestion pipeline must become a CLI script before Phase 1**
The only current way to build the knowledge base is to run a Jupyter notebook. A headless App VM has no Jupyter. This needs to be converted to a runnable Python script before Phase 1 deployment is possible.

**PrimePack AG branding and system prompts in three places**
All three are hardcoded and need to be adapted to the actual use case before any real deployment:

| Location | Content |
|----------|---------|
| `frontend/src/config.ts` | UI branding — company name and agent name ("Prime") |
| `backend/src/.../main.py:60` | Full production system prompt — 5 rules, role definition, all referencing PrimePack AG sustainability compliance |
| `backend/src/.../feature0_baseline_rag.py:76` | Shorter system prompt used when running the feature script standalone — also references PrimePack AG |

The system prompts in particular encode assumptions about the domain (sustainability/EPDs/supplier claims) that won't apply to a different use case. They should be moved to an env var or external config file so they can be changed per deployment without touching code.

**The `rlei_deduplication` branch contains custom chunking work — and the upstream deduplication is broken for incremental updates**
The current ingestion pipeline has a very coarse guard: if the ChromaDB collection contains *any* chunks, the entire embedding run is skipped (`feature0_baseline_rag.py:269`). There is no per-document or content-hash deduplication. Consequences:
- Running ingestion on an empty store works correctly
- Adding a new document to an existing store → the new document silently never gets indexed
- Re-ingesting a renamed file → duplicate content enters the store

This is almost certainly what the `rlei_deduplication` branch addresses. Decide before Phase 1 whether to merge it in — deploying with the upstream ingestion and switching later invalidates any already-indexed vector store and requires a full re-ingestion from scratch.

**Ollama model download in a corporate network**
Ollama downloads models from HuggingFace on first run. In a corporate environment with restricted internet access this will likely be blocked. The model needs to be pre-downloaded and served from a local path, or a local HuggingFace mirror needs to be configured. Plan this with IT before setting up the inference VM.

### Open questions for Phase 2

- Final LLM model choice — benchmark Gemma 4 26B A4B MoE vs 31B Dense on the Spark once hardware is available; MoE is the current recommendation
- Is a single shared knowledge base acceptable, or do departments need isolated vector stores?
- Who manages the ingestion pipeline — IT, or knowledge owners directly?
- Should the ingestion pipeline have a management UI, or is a CLI/script sufficient for Phase 2?

### Resolved

- **Concurrent users:** Max ~5 concurrent, 40 total users — well within vLLM's capacity with the recommended MoE model
- **Auth:** Local Active Directory via AD FS (preferred); Entra ID as future fallback
- **LLM serving:** vLLM (Phase 2 / Spark), Ollama (Phase 1 / CPU VM)
- **Data residency:** Fully on-premises — no data leaves the company network in either phase
