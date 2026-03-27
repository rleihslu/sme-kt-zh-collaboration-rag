# SME-KT-ZH Collaboration: RAG Prototyp

A collaborative prototyping project building an open-source RAG (Retrieval-Augmented Generation) backbone that SMEs can adapt for their own knowledge management.

## Scenario
**PrimePack AG** sells packaging products (pallets, cardboard boxes, tape) sourced from multiple suppliers. Sustainability claims are increasingly important, from customers requiring CSRD-compliant data to procurement needing to evaluate supplier evidence quality.

The core challenge: sustainability information is spread across many documents (EPDs, supplier brochures, company reports, regulatory frameworks), claims have widely varying evidence quality, and employees must answer questions like:

- *"What does Supplier A claim for Product X, and can we trust it?"*
- *"Which suppliers are compliant with our EPD requirement?"*
- *"Can we tell a customer that this tape is PFAS-free?"*
- *"Which evidence is missing before we can respond?"*

The RAG assistant should **cite sources explicitly**, **separate facts from claims**, **highlight gaps and conflicts**, and **refuse to conclude when evidence is insufficient**.

---

## Installation

Requires Python version >=3.12

```bash
# Create and activate a virtual environment
python -m venv rag_venv
source rag_venv/bin/activate  # on Windows: rag_venv\Scripts\activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

The `conversational_toolkit` and `backend` packages are installed automatically via `requirements.txt`. If needed, install them manually:

```bash
pip install -e conversational-toolkit/
pip install -e backend/
```

### Ollama (local LLM)

[Ollama](https://ollama.com) lets you run open-weight language models locally -> no API key, no data sent externally. Use it as a privacy-preserving alternative to the OpenAI backend.

**Install Ollama**

| Platform | Command                                                                                  |
|----------|------------------------------------------------------------------------------------------|
| macOS    | `brew install ollama` or download the app from [ollama.com](https://ollama.com/download) |
| Linux    | `curl -fsSL https://ollama.com/install.sh \| sh`                                         |
| Windows  | Download the installer from [ollama.com](https://ollama.com/download)                    |

**Start the server and download a model**

```bash
# Start the Ollama server (runs in the background on port 11434)
ollama serve

# Pull the default workshop model (~7 GB)
ollama pull mistral-nemo:12b

# List models already downloaded
ollama list
```

The server must be running whenever you use `BACKEND=ollama`. On macOS the desktop app starts the server automatically; on Linux run `ollama serve` in a separate terminal.

---

## Running the Notebooks (JupyterLab)

The workshop notebooks require JupyterLab. You can run them in an IDE or start it from the project root after activating your virtual environment:

```bash
jupyter lab
```

JupyterLab opens in your browser at `http://localhost:8888`. Navigate to `backend/notebooks/` in the file browser on the left and open `feature0a_baseline_rag.ipynb` to start.

**Running cells:** Use `Shift+Enter` to run a cell and advance to the next one. Run cells top to bottom, each notebook must be executed in order within a session.

**OpenAI API key:** If you want to use OpenAI models, you need to set an OpenAI API key. Set the key before starting JupyterLab, or export it inside a cell:

```bash
export OPENAI_API_KEY="sk-..."
jupyter lab
```

---

## Quick Start

### Running the full stack (backend API + frontend)

This launches the FastAPI backend and the Next.js chat frontend together.

**Prerequisites:** Python ≥ 3.12, Node.js / npm, and either an OpenAI API key or a running Ollama server (see [Ollama](#ollama-local-llm) below).

**Step 1: Install Python dependencies** (skip if already done)

```bash
python -m venv rag_venv
source rag_venv/bin/activate  # on Windows: rag_venv\Scripts\activate
pip install -r requirements.txt
```

**Step 2: Create vector store**

Run `backend/notebooks/demo_build_db.ipynb` once to populate the shared vector stores. It might take aome time to create the image vector store.

**Step 3: Create `.env.local`** (required for `make run`)

The Makefile reads your API key from a `.env.local` file in the project root. Create it with:

```bash
echo "OPENAI_API_KEY=sk-..." > .env.local
```

Or open the file in an editor and add:

```
OPENAI_API_KEY=sk-...
```

> If you run the backend without the Makefile, you can export the key in your shell instead and skip this file — see the "Without the Makefile" command in Step 3.

**Step 4: Start the backend**

With the Makefile:

```bash
make run
```

Without the Makefile:

```bash
# OpenAI
BACKEND=openai python -m sme_kt_zh_collaboration_rag.main

# Ollama (requires `ollama serve` and a pulled model)
BACKEND=ollama python -m sme_kt_zh_collaboration_rag.main
```

The server starts at `http://localhost:8080`.

**Step 5: Start the frontend** (in a separate terminal)

```bash
cd frontend
npm install # first time only
npm run dev
```

**Step 6: Open the app**

Go to `http://localhost:3000` in your browser.

---

### Notebooks (exploration and learning)

```bash
jupyter lab
# Open backend/notebooks/feature0a_baseline_rag.ipynb to start
```

### Single query from the command line

```bash
BACKEND=openai python -m sme_kt_zh_collaboration_rag.feature0_baseline_rag
BACKEND=openai QUERY="Which tape products have a verified EPD?" python -m sme_kt_zh_collaboration_rag.feature0_baseline_rag
BACKEND=openai RESET_VS=1 python -m sme_kt_zh_collaboration_rag.feature0_baseline_rag  # rebuild vector store
```

---

## Workshop Feature Tracks

The workshop is structured as progressive feature tracks, each implemented as one or more Jupyter notebooks in `backend/notebooks/`. Run notebooks top to bottom within a session.

Run `demo_build_db.ipynb` once before the feature notebooks to populate the shared vector stores. Feature notebooks import the `conversational_toolkit` library and expect those stores to exist.

| Notebook                                                 | Topic                            | What you learn                                                                                                                                                                    |
|----------------------------------------------------------|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `demo_build_db.ipynb`                                    | **Setup: Build Vector Stores**   | Runs once to embed the full corpus and persist two ChromaDB collections: `vs_text` (OpenAI embeddings) and `vs_image` (Qwen3-VL embeddings) under `backend/src/sme_kt_zh_collaboration_rag/db/`. |
| `feature0a_baseline_rag.ipynb`                           | **Baseline RAG Pipeline**        | The five-stage RAG loop (chunk, embed, store, retrieve, generate), how retrieval inspection works as a debugging tool, and the main failure modes the workshop addresses.         |
| `feature0b_ingestion.ipynb`                              | **Document Ingestion**           | PDF parser comparison (markitdown vs. docling), chunking strategies (fixed-size, header-based, paragraph-aware), token limits, and how chunking choice affects retrieval quality. |
| `feature1a_evaluation.ipynb`                             | **Evaluation & Validation**      | Reference-free RAGAS metrics (Faithfulness, AnswerRelevancy); ground-truth metrics (ContextPrecision, ContextRecall); systematic comparison of RAG pipeline configurations.       |
| `feature1b_dataset_creation.ipynb`                       | **Synthetic Dataset Generation** | Building a validation dataset from vector store content; LLM-generated Q&A pairs; scaling beyond manual ground-truth creation.                                                    |
| `feature2a_structured_outputs.ipynb`                     | **Structured Outputs**           | Entity-grounding prompts to prevent hallucination; structured JSON outputs with evidence levels (VERIFIED / CLAIMED / MISSING / MIXED); making claim quality machine-readable.    |
| `feature3a_advanced_retrieval.ipynb`                     | **Retrieval Strategies**         | BM25 keyword retrieval; hybrid search via Reciprocal Rank Fusion (RRF); metadata filtering to scope retrieval to known documents.                                                 |
| `feature3b_conversation.ipynb`                           | **Conversational RAG**           | Maintaining conversation history; multi-turn context management.                                                                                                                  |
| `feature3c_query_techniques.ipynb`                       | **Query Improvement**            | Query expansion and HyDE (Hypothetical Document Embeddings) to improve semantic retrieval.                                                                                        |
| `feature3d_context_enrichment.ipynb`                     | **Context Enrichment**           | Neighbouring-chunk expansion to handle answers that span chunk boundaries; token budget analysis.                                                                                 |
| `feature4a_tools.ipynb` – `feature4e_rag_subagent.ipynb` | **Tools & Agents**               | LLM tool use; RAG as a tool; multi-step agent workflows; supplier comparison and evidence audit patterns.                                                                         |

---

## Repository Structure

```
.
├── backend/                                # RAG pipeline application
│   ├── notebooks/                          # Workshop notebooks
│   │   ├── demo_build_db.ipynb             # One-off setup: build text + image vector stores
│   │   ├── feature0a_baseline_rag.ipynb
│   │   ├── feature0b_ingestion.ipynb
│   │   ├── feature1a_evaluation.ipynb
│   │   ├── feature1b_dataset_creation.ipynb
│   │   ├── feature2a_structured_outputs.ipynb
│   │   ├── feature3a_advanced_retrieval.ipynb
│   │   ├── feature3b_conversation.ipynb
│   │   ├── feature3c_query_techniques.ipynb
│   │   ├── feature3d_context_enrichment.ipynb
│   │   └── feature4a_tools.ipynb – feature4e_rag_subagent.ipynb
│   └── src/sme_kt_zh_collaboration_rag/
│       ├── db/
│       │   ├── vs_text/                    # ChromaDB collection: text chunks (OpenAI embeddings)
│       │   └── vs_image/                   # ChromaDB collection: image chunks (Qwen3-VL embeddings)
│       ├── feature0_baseline_rag.py        # Five-step pipeline (chunking, embedding, retrieval, generation)
│       ├── feature0_ingestion.py           # Parser comparison, chunking utilities, token analysis
│       ├── feature1_evaluation.py          # Shared evaluation queries and ground-truth answers
│       ├── feature3_advanced_retrieval.py
│       └── main.py                         # FastAPI server entry point
│
├── conversational-toolkit/                 # Reusable RAG components (toolkit library)
│   └── src/conversational_toolkit/
│       ├── agents/                         # RAG agent (retrieval + generation)
│       ├── api/                            # FastAPI server and routes
│       ├── chunking/                       # PDF, Excel, Markdown chunkers
│       ├── embeddings/                     # Sentence Transformer embeddings
│       ├── llms/                           # OpenAI, Ollama, local LLM backends
│       ├── retriever/                      # Vector store retriever, BM25, hybrid
│       ├── utils/                          # Prompt building, query expansion, RRF
│       └── vectorstores/                   # ChromaDB vector store
│
├── data/                                   # Document corpus
│
└── frontend/                               # Next.js chat frontend
```

---

## Dataset

All documents are in `data/`. The file prefix defines the document type:

| Prefix        | Meaning                                                        |
|---------------|----------------------------------------------------------------|
| `ART_`        | Artificial workshop scenario document                          |
| `EPD_`        | Verified, third-party Environmental Product Declaration        |
| `SPEC_`       | Product specification, datasheet, or article document          |
| `REF_`        | Regulatory or reference document                               |
| `EVALUATION_` | Evaluation / ground-truth dataset (not part of the RAG corpus) |

---

### Artificial Documents (`ART_`)

Designed with deliberate flaws to demonstrate RAG failure modes.

| File | Type | What it demonstrates |
|--------|--------|--------|
| `ART_product_catalog.pdf` | Internal policy | Portfolio scope, evidence-level policy, customer communication rules, open action items. The machine-readable product table lives in `product_overview.xlsx`. |
| `ART_internal_procurement_policy.pdf` | Internal policy | Evidence levels (A–D), EPD requirements, PFAS policy |
| `ART_supplier_brochure_tesa_ECO.pdf` | Supplier marketing | **Unverified claim**: 68% CO₂ reduction, no EPD |
| `ART_supplier_brochure_CPR_wood_pallet.pdf` | Supplier marketing | **Unverified LCA figures**; FSC claim without certificate |
| `ART_logylight_incomplete_datasheet.pdf` | Supplier datasheet | **Missing data**: all LCA fields "not yet available" |
| `ART_relicyc_logypal1_old_datasheet_2021.pdf` | Superseded document | **Temporal conflict**: 2021 figure vs. 2023 verified EPD |
| `ART_customer_inquiry_frische_felder.pdf` | Customer communication | CSRD request with open [ACTION] items |

---

### Environmental Product Declarations (`EPD_`)

Third-party verified EPDs. Naming: `EPD_{category}_{supplier}_{product}.pdf`

**Pallets:**

| File                                | Supplier      | Product (ID)              |
|-------------------------------------|---------------|---------------------------|
| `EPD_pallet_CPR_noe.pdf`            | CPR System    | Noé Pallet (32-100)       |
| `EPD_pallet_relicyc_logypal1.pdf`   | Relicyc       | Logypal 1 (32-103) — 2023 |
| `EPD_pallet_stabilplastik_ep08.pdf` | StabilPlastik | EP 08 (32-105)            |

**Cardboard Boxes:**

| File                                    | Supplier | Product (ID)                  |
|-----------------------------------------|----------|-------------------------------|
| `EPD_cardboard_redbox_cartonpallet.pdf` | Redbox   | Cartonpallet CMP (11-100)     |
| `EPD_cardboard_grupak_corrugated.pdf`   | Grupak   | Corrugated cardboard (11-101) |

**Tape:**

| File                              | Supplier | Product (ID)                  |
|-----------------------------------|----------|-------------------------------|
| `EPD_tape_IPG_hotmelt.pdf`        | IPG      | Hot Melt Tape (50-100)        |
| `EPD_tape_IPG_wateractivated.pdf` | IPG      | Water-Activated Tape (50-101) |

---

### Product Specifications & Datasheets (`SPEC_`)

Supplier-provided specs, datasheets, and article documents. Naming: `SPEC_{category}_{supplier}_{product}.pdf`

**Pallets:**

| File                                   | Supplier      | Product                      |
|----------------------------------------|---------------|------------------------------|
| `SPEC_pallet_CPR_plastic.pdf`          | CPR System    | Plastic Pallet (32-102)      |
| `SPEC_pallet_CPR_wood.pdf`             | CPR System    | Wooden Pallet (32-101)       |
| `SPEC_pallet_CPR_recycled_plastic.pdf` | CPR System    | Recycled plastic pallet PR12 |
| `SPEC_pallet_relicyc_logypal1.pdf`     | Relicyc       | Logypal 1: product info     |
| `SPEC_pallet_relicyc_1208MR.pdf`       | Relicyc       | Pallet 1208MR                |
| `SPEC_pallet_stabilplastik_ep08.pdf`   | StabilPlastik | EP 08: datasheet            |

**Tape:**

| File                                            | Supplier | Product                                     |
|-------------------------------------------------|----------|---------------------------------------------|
| `SPEC_tape_CST_synthetic_rubber.pdf`            | CST      | Synthetic rubber tape                       |
| `SPEC_tape_WAT_central_brand.pdf`               | WAT      | WAT tape: article document                 |
| `SPEC_tape_WAT_260.pdf`                         | WAT      | WAT 260: technical data sheet              |
| `SPEC_tape_tesa_tesapack58297.pdf`              | Tesa     | tesapack 58297 (German)                     |
| `SPEC_tape_tesa_sustainability_report_2024.pdf` | Tesa     | Company sustainability report 2024 (German) |

**Master data:**

| File                        | Content                                                                                                                                                            |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ART_product_overview.xlsx` | Machine-readable product table: all 12 products with IDs, suppliers, categories, and EPD status. Referenced by `ART_product_catalog.md` for the full product list. |

---

### Reference & Regulatory Documents (`REF_`)

| File                                         | Content                                                |
|----------------------------------------------|--------------------------------------------------------|
| `REF_ghg_protocol_product_lca.pdf`           | GHG Protocol: Product LCA standard                    |
| `REF_ghg_protocol_corporate_value_chain.pdf` | GHG Protocol: Scope 3 / value chain                   |
| `REF_ghg_protocol_corporate_standard.pdf`    | GHG Protocol: revised corporate standard              |
| `REF_eu_csrd.pdf`                            | EU Corporate Sustainability Reporting Directive (CSRD) |
| `REF_eu_climate_reporting_guidelines.pdf`    | EU guidelines on climate-related reporting             |
| `REF_ch_eu_sustainability_obligations.pdf`   | Swiss/EU sustainability obligations                    |
| `REF_iso14024_conformance_statement.pdf`     | ISO 14024 conformance statement                        |
| `REF_ecologo_catalogue.pdf`                  | ecoLogo certification catalogue                        |

---

### Evaluation Document

| File                            | Content                                                                                                                                              |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `EVALUATION_qa_ground_truth.md` | Sample questions with expected answers, required sources, and difficulty ratings. Use to manually verify RAG output. Not included in the RAG corpus. |

---

## Key Design Decisions in the Dataset

**Evidence quality varies deliberately.** Products span verified EPDs (Level A) to missing data (Level D). The RAG must distinguish these, not just retrieve numbers.

**Conflicts are intentional.** `ART_relicyc_logypal1_old_datasheet_2021.md` and `EPD_pallet_relicyc_logypal1.pdf` contain different GWP figures. The RAG should flag the conflict and prefer the more recent, verified source.

**Gaps are part of the answer.** For the LogyLight pallet and PFAS declarations, the correct answer is "we don't have this data." A RAG that fabricates an answer here fails.

**The portfolio boundary matters.** The product catalog defines which products exist. Queries about products outside the catalog (e.g., "Lara Pallet") should be answered as "not in our portfolio."

---

## LLM Backends

`BACKEND` must always be set explicitly, there is no default.

| Backend | Command | Notes |
|-----|-----|-----|
| Ollama | `BACKEND=ollama python -m sme_kt_zh_collaboration_rag.feature0_baseline_rag` | Requires `ollama serve` and `ollama pull mistral-nemo:12b` |
| OpenAI  | `BACKEND=openai python -m ...` | Requires `OPENAI_API_KEY` env var or `/secrets/OPENAI_API_KEY` file |

Override the model: `MODEL=gpt-4o BACKEND=openai python -m ...`

---

## Contributing

We welcome contributions: new feature notebooks, improvements to the toolkit library, additional documents in the dataset, or bug fixes. The steps below walk you through the full workflow from cloning to an approved pull request.

### 1. Clone the repository

```bash
git clone https://github.com/kanton-zurich/sme-kt-zh-collaboration-rag.git
cd sme-kt-zh-collaboration-rag
```

### 2. Set up your environment

```bash
# Create and activate a virtual environment
python -m venv rag_venv
source rag_venv/bin/activate  # on Windows: verag_venvnv\Scripts\activate

# Install all dependencies including the two editable packages
pip install -r requirements.txt

# Install pre-commit hooks (runs formatters and linters before every commit)
pre-commit install
```

### 3. Create a feature branch

Always work on a dedicated branch, never directly on `main`.

```bash
git checkout -b feature/my-descriptive-branch-name
```

Branch naming conventions:

- `feature/`: new functionality or notebooks
- `fix/`: bug fixes
- `docs/`: documentation-only changes
- `data/`: additions or corrections to the document corpus

### 4. Make your changes

A few guidelines depending on what you are changing:

**Notebooks (`backend/notebooks/`):**
- Keep cells focused and commented.
- Follow the existing naming convention: `feature{N}_{short_topic}.ipynb`.

**Toolkit library (`conversational-toolkit/`):**
- Add or update unit tests in the corresponding `tests/` directory.
- Run `pytest` before committing to confirm nothing is broken.
- Keep new classes and functions typed (the project uses `mypy`).

**Dataset (`data/`):**
- Follow the file-prefix convention: `EPD_`, `SPEC_`, `REF_`, `ART_`, or `EVALUATION_`.
- If adding a document that the RAG corpus should index, also add an entry to `README.md` under the appropriate `Dataset` subsection.
- If a new document introduces a deliberate flaw (conflict, missing data, unverified claim), describe it in the table.

### 5. Verify your changes locally

```bash
# Run the test suite
pytest
```

Fix any reported issues before continuing.

### 6. Commit your changes
Write concise commit messages (e.g. `fix chunker token-limit handling`, `add EPD for StabilPlastik EP08`).

```bash
git add path/to/changed/files
git commit -m "short description of what and why"
```

### 7. Push and open a pull request

```bash
git push -u origin feature/my-descriptive-branch-name
```

Then open a pull request on GitHub:

1. Go to the repository on GitHub and click **"Compare & pull request"** (the banner that appears after you push).
2. Set the **base branch** to `main`.
3. Write a clear PR description:
    - **What** does this PR change?
    - **Why** is the change needed or useful?
    - **How** was it tested (notebook run-through, `pytest`, manual check)?
    - If it closes an open issue, add `Closes #<issue-number>` in the description.
4. Request a review from at least one other contributor.

### 8. Respond to review feedback

Address comments by pushing additional commits to the same branch, do not open a new PR. Once all review threads are resolved and the CI checks pass, a maintainer will merge your PR into `main`.

---

## Development Reference

```bash
# Run tests
pytest

# Clear notebook outputs (replace with your notebook path)
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace backend/notebooks/feature0a_baseline_rag.ipynb
```
