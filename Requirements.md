# ExtractMark — Requirements

**Project:** ExtractMark — Self-Hosted Document Extraction Model Evaluation
**Platform:** NVIDIA DGX Spark (ARM64, Ubuntu) — personal workstation
**Constraint:** No external APIs — fully on-premise inference; personal use only; public datasets exclusively
**Author:** Kashif Muhammad
**Date:** 2026-04-03
**Status:** Draft v1.5

---

## 1. Project Overview

ExtractMark benchmarks multiple self-hostable document extraction models on a personal NVIDIA DGX Spark workstation. The goal is to identify the optimal model or pipeline combination for document extraction tasks — evaluating candidates that could later be integrated into enterprise knowledge graph ingestion pipelines processing 1000+ documents (PDF, Word, PPT) under on-premise constraints.

The evaluation covers text accuracy, table fidelity, layout understanding, bounding box precision, and reading order correctness using a **multi-layer metric framework** (edit distance, semantic similarity, binary unit tests, and LLM-as-judge) to capture error classes that any single metric misses. Results will inform architectural decisions for downstream RAG and knowledge graph construction.

> **Note:** This benchmark is conducted for personal research and learning purposes on the author's home DGX Spark. No corporate documents are used — evaluation is exclusively on public benchmark datasets. All models are evaluated under their respective licences for personal/research use.

Models are deployed via **vLLM** (OpenAI-compatible API, GPU-optimised), the sole benchmarking runtime for this project. vLLM is supported on DGX Spark ARM64/Linux and provides the production-grade serving path required for pipeline integration.

In addition to VLM/LLM models, the benchmark also covers **Python parsing libraries** (Section 5). Libraries and models serve different layers of the extraction stack and are benchmarked independently before being composed into candidate pipelines.

---

## 2. Goals

- **G-01** Evaluate all shortlisted models under identical conditions on the DGX Spark
- **G-02** Produce comparable, reproducible benchmark scores across all models
- **G-03** Identify the best model(s) for enterprise document extraction (PPT, Word, PDF — technical/automotive domain)
- **G-04** Validate spatial grounding (bounding boxes) capability for knowledge graph node/edge extraction
- **G-05** Provide a clear recommendation with supporting evidence for pipeline integration
- **G-06** Identify the optimal library or library combination per document format as the ingestion layer beneath the VLM extraction stack
- **G-07** Evaluate extraction quality using a multi-layer metric framework (edit distance, semantic similarity, binary unit tests, LLM-as-judge) to capture error classes that any single metric misses

---

## 3. Scope

### 3.1 In Scope

- Self-hosted model inference (no external API calls)
- Evaluation of Python parsing libraries (rule-based, heuristic, and AI-augmented pipelines)
- Evaluation on public benchmark datasets (OmniDocBench v1.5, OlmOCR-Bench, DocLayNet, FinTabNet, FUNSD, DocVQA)
- Multi-layer quantitative evaluation: edit distance (CER/WER), semantic similarity (SBERT), binary unit tests (OlmOCR-Bench), and LLM-as-judge
- Adoption of OmniDocBench v1.5 and OlmOCR-Bench evaluation frameworks (reuse existing eval code rather than custom scripts)
- Output normalisation to canonical Markdown before metric evaluation
- Deployment via vLLM (OpenAI-compatible API) as the sole benchmarking runtime
- ARM64 / Linux compatibility verification per model

### 3.2 Out of Scope

- Fine-tuning or re-training of any model
- Cloud or SaaS-based OCR solutions (Mistral OCR 3, Azure Document Intelligence, etc.)
- Integration into a full enterprise knowledge graph pipeline (separate project phase)
- Real-time / streaming inference optimisation
- Multi-GPU distributed inference (single GPU per model run unless required)

---

## 4. Models Under Evaluation

### 4.1 Deployment Runtime

All models are benchmarked exclusively via **vLLM** — GPU-optimised, OpenAI-compatible API serving. vLLM supports custom architectures (encoder-decoder, non-GGUF) and provides the production-grade throughput measurement required for pipeline integration decisions.

### 4.2 NVIDIA Nemotron — Important Distinction

The Nemotron family contains two distinct model types relevant to this project:

| Nemotron Variant | Purpose | Suitable For |
|---|---|---|
| **Nemotron Parse v1.1 / v1.2** | Vision-encoder-decoder OCR model | Document extraction frontend (M-01) |
| **Nemotron-3-Nano (30B MoE)** | General reasoning LLM | Downstream extraction LLM backbone (L-02) |

Nemotron Parse is the OCR frontend model under evaluation. Nemotron-3-Nano is an LLM (not an OCR model) and is evaluated separately as a potential replacement for Qwen2.5-7B in the table-based extraction pipeline stage.

### 4.3 OCR Frontend Models

| ID | Model | Params | vLLM | Bbox | Licence | Notes |
|----|-------|--------|------|------|---------|-------|
| M-01 | **Nemotron Parse v1.1** | <1B | ✅ | ✅ | NVIDIA Open Model | Primary OCR candidate; custom mBart encoder-decoder |
| M-02 | **LightOnOCR-2-1B-bbox-soup** | ~1B | ✅ | ✅ | Apache 2.0 | Fastest; stability tuning required on table-heavy docs |
| M-03 | **GOT-OCR 2.0** | ~580M | ✅ | Partial | Apache 2.0 | HTML/LaTeX/Markdown output; transformers path available |
| M-04 | **OlmOCR-2** | ~7B | ✅ | ❌ | Apache 2.0 | Open training data; English-optimised |
| M-05 | **Docling + PaddleOCR** | Pipeline | N/A | Partial | Apache 2.0 | Current baseline; native Python; no vision for figures |
| M-06 | **GLM-OCR** | 0.9B | ✅ | ✅ | MIT | **#1 OmniDocBench V1.5 (94.62)**; SDK included |
| M-07 | **Qwen2.5-VL-7B** | 7B | ✅ | ✅ | Apache 2.0 | Strong doc understanding + bounding boxes |
| M-08 | **Qwen2.5-VL-3B** | 3B | ✅ | ✅ | Apache 2.0 | Edge-efficient variant; compare throughput vs 7B |
| M-09 | **InternVL2.5-4B** | 4B | ✅ | ✅ | Apache 2.0 | 300M InternViT + 3B Qwen2.5 LM; 448×448 tile dynamic resolution |
| M-10 | **OCRFlux-3B** | 3B | ✅ | ❌ | Apache 2.0 | Qwen2.5-VL-3B fine-tuned; cross-page table + paragraph merging |
| M-11 | **DeepSeek-OCR** | ~3B | ✅ | ✅ | MIT | Grounding mode via `<\|grounding\|>` prompt |
| M-12 | **Chandra 2** (Datalab) | ~4B | ✅ | ✅ | OpenRAIL-M | **#1 olmOCR-Bench (85.9%)**; Markdown/HTML/JSON output; from Surya/Marker team; personal use permitted under licence |
| M-13 | **PaddleOCR-VL** | ~0.9B | ✅ | ✅ | Apache 2.0 | Sub-1B single-pass VLM; ~79–80% olmOCR-Bench; lightweight throughput baseline |

### 4.4 LLM Backbone Candidates (Pipeline Extraction Stage Only)

These are not OCR frontend models. They are evaluated only as the downstream extraction LLM in a table-based or replace-based pipeline (replacing Qwen2.5-7B from arxiv 2510.10138). Tested independently of the OCR frontend benchmark.

| ID | Model | Params | vLLM | Notes |
|----|-------|--------|------|-------|
| L-01 | **Qwen2.5-7B** | 7B | ✅ | Current baseline from arxiv 2510.10138 paper |
| L-02 | **Nemotron-3-Nano (30B)** | 30B MoE | ✅ | Reasoning-capable; configurable thinking budget |
| L-03 | **Qwen2.5-VL-72B** | 72B | ✅ | High-accuracy; **Q4 quantisation mandatory** on DGX Spark; schedule last |

### 4.5 vLLM Serve Reference

All models are served via `vllm serve` on localhost. Example commands (adjust `--model` and `--dtype` per model):

```bash
# OCR frontend models
vllm serve nvidia/nemotron-parse-v1.1 --dtype bfloat16                 # M-01
vllm serve lightonai/LightOnOCR-2-1B-bbox-soup --dtype bfloat16       # M-02
vllm serve ucaslcl/GOT-OCR2_0 --dtype bfloat16                        # M-03
vllm serve allenai/olmocr2-7b --dtype bfloat16                         # M-04
vllm serve THUDM/glm-ocr-2b --dtype bfloat16                          # M-06
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16               # M-07
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --dtype bfloat16               # M-08
vllm serve OpenGVLab/InternVL2_5-4B --dtype bfloat16                   # M-09
vllm serve Aryn/OCRFlux-3B --dtype bfloat16                            # M-10
vllm serve deepseek-ai/DeepSeek-OCR --dtype bfloat16                   # M-11
vllm serve datalab-to/chandra-ocr-2 --dtype bfloat16                   # M-12
vllm serve PaddlePaddle/PaddleOCR-VL --dtype bfloat16                  # M-13

# LLM backbone candidates
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype bfloat16                  # L-01
vllm serve nvidia/Nemotron-3-Nano-30B --dtype bfloat16                 # L-02
vllm serve Qwen/Qwen2.5-VL-72B-Instruct --dtype bfloat16 --quantization awq  # L-03 (Q4 mandatory)
```

> **Note:** Model HuggingFace IDs above are indicative — verify exact repo names before first pull. Models requiring quantisation (L-03) should use `--quantization awq` or `--quantization gptq` as appropriate.

---

## 5. Libraries Under Evaluation

### 5.1 Extraction Stack Architecture

Libraries and VLM models serve distinct layers and are benchmarked independently before being composed into candidate pipelines:

```
Layer 1 — Format ingestion:    python-docx / python-pptx / pdf2image / PyMuPDF
Layer 2 — Layout + text:       Docling / MinerU / marker / Unstructured / MarkItDown
Layer 3 — OCR (scanned docs):  PaddleOCR / Surya / Tesseract / VLM model
Layer 4 — Visual elements:     VLM model (Nemotron Parse / GLM-OCR / Qwen2.5-VL)
Layer 5 — Extraction LLM:      Qwen2.5-7B / Nemotron-3-Nano
```

Libraries in this section cover Layers 1–3. VLM models (Section 4.3) cover Layers 3–4. LLM backbones (Section 4.4) cover Layer 5.

### 5.2 Tier 1 — Rule-Based / Heuristic Libraries (CPU, deterministic, no GPU)

These form the baseline ingestion layer and are benchmarked for speed, text fidelity, and table accuracy on digital (non-scanned) documents.

| ID | Library | Format | Strength | Weakness | Notes |
|----|---------|--------|----------|----------|-------|
| LIB-01 | **PyMuPDF / pymupdf4llm** | PDF | Fastest text extraction (~0.12s/page); native table detection; Markdown via `pymupdf4llm` | No OCR; struggles on multi-column | De facto baseline for digital PDFs; used in MinerU internals |
| LIB-02 | **pdfplumber** | PDF | Coordinate-level layout control; reliable lattice/stream table extraction | Needs config tuning; no OCR | Built on pdfminer.six; strong for structured financial/tabular PDFs |
| LIB-03 | **pypdfium2** | PDF | Blazing speed (~0.003s/page); consistent word order | No structure, no tables, no OCR | Use as raw throughput baseline only |
| LIB-04 | **Camelot** | PDF | Best table extraction for bordered tables (lattice mode); DataFrame output | Fails on borderless/nested tables; requires Ghostscript | Renamed to `pypdf_table_extraction`; pair with PyMuPDF for full pipeline |
| LIB-05 | **Tabula-py** | PDF | Table extraction via Java PDFBox; stream mode for borderless tables | Requires JVM; slower than Camelot | Better than Camelot for tables without visible borders |
| LIB-06 | **python-docx** | DOCX | Direct Word object model (paragraphs, runs, styles, tables); deterministic | DOCX only | Essential baseline for Word document extraction |
| LIB-07 | **python-pptx** | PPTX | Native PPT slide/shape/table/text-frame access | PPTX only | Only library with native PPTX support; critical for slide corpus extraction |

### 5.3 Tier 2 — AI-Augmented Pipeline Libraries (GPU-optional or GPU-beneficial)

These combine heuristics with ML models to handle complex layouts, scanned content, and multi-format ingestion.

| ID | Library | Format | Strength | Weakness | Notes |
|----|---------|--------|----------|----------|-------|
| LIB-08 | **Docling** (IBM) | PDF, DOCX, PPTX, HTML | Layout-aware; reading order; TableFormer for tables; HuggingFace native; NVIDIA-partnered (4× GPU speedup on DGX Spark) | No figure/image interpretation | Current baseline in M-05; also standalone library benchmark |
| LIB-09 | **MinerU** | PDF only | Hybrid rule+model; rotated table detection; header/footer removal; strong on Chinese/financial/scientific docs | PDF only; HTML table output; no native page-split | Used in Doc-Researcher architecture; strong complex layout recovery |
| LIB-10 | **marker** | PDF, DOCX, PPTX, images | Surya OCR + layout detection; Markdown output; GPU-accelerated; by creator of Surya | Multi-column layouts can scramble; newer, smaller community | Good mid-point between speed and accuracy; benchmark vs Docling |
| LIB-11 | **Surya** | Images / PDF pages | Standalone layout detection + line-level OCR; modular; 90+ languages; outperforms Tesseract on most benchmarks; by creator of marker | GPU preferred; no table structure | Test independently as a layout-only layer before pairing with extraction LLM |
| LIB-12 | **Unstructured** | PDF, DOCX, PPTX, HTML, images | Semantic element partitioning (title, para, table, caption); RAG-ready chunks; LangChain/LlamaIndex native | Slower (~1.3s/page); scanned PDFs still need OCR backend | Best multi-format semantic chunker; evaluate RAG output quality vs Docling |
| LIB-13 | **MarkItDown** (Microsoft) | PDF, DOCX, PPTX, XLSX, HTML | Zero-dependency Markdown conversion; lightweight; already used in arxiv 2510.10138 as fast-path | Lossy — no spatial structure or bounding boxes | Benchmark as cheap fast-path baseline; already validated in table-based pipeline |

### 5.4 Tier 3 — Specialised / Supplementary

Evaluated selectively for specific element types or as fallback components.

| ID | Library / Tool | Purpose | Notes |
|----|---------------|---------|-------|
| LIB-14 | **Tesseract** (via `pytesseract`) | Classic CPU OCR engine | Benchmark as lowest-cost scanned-doc fallback; compare vs PaddleOCR on scanned PDFs |
| LIB-15 | **Table Transformer (TATR)** | Table detection + structure (transformer-based) | Outperforms all rule-based tools on Scientific/Financial recall; pairs with any text extractor as table-specialist layer |
| LIB-16 | **Nougat** (Meta) | Scientific PDF parsing | Encoder-decoder trained on academic docs; strong on equations; weak on non-academic; test only on D-01 scientific subset |
| LIB-17 | **pdfminer.six** | Low-level PDF text + layout glyphs | PyMuPDF and pdfplumber both build on this; test only if glyph-level control is required for edge cases |

### 5.5 Library × Format Matrix

Recommended library combinations per document scenario:

| Scenario | Recommended Combination |
|----------|------------------------|
| Digital PDF — fast baseline | LIB-01 (PyMuPDF / pymupdf4llm) |
| Digital PDF — table-heavy | LIB-01 + LIB-04 (Camelot lattice) or LIB-02 (pdfplumber) |
| Scanned PDF — no GPU | LIB-01 + LIB-14 (Tesseract) |
| Scanned PDF — GPU available | LIB-09 (MinerU) or LIB-10 (marker) |
| Complex multi-column PDF | LIB-08 (Docling) or LIB-09 (MinerU) |
| DOCX — structured extraction | LIB-06 (python-docx) + LIB-12 (Unstructured) |
| PPTX — slide extraction | LIB-07 (python-pptx) + LIB-12 (Unstructured) |
| Multi-format RAG chunking | LIB-08 (Docling) or LIB-12 (Unstructured) |
| Fast Markdown conversion | LIB-13 (MarkItDown) |
| Tables on scientific/financial docs | LIB-15 (TATR) as specialist layer |

### 5.6 Install Reference

```bash
# Tier 1 — Rule-based
pip install pymupdf pymupdf4llm          # LIB-01
pip install pdfplumber                    # LIB-02
pip install pypdfium2                     # LIB-03
pip install camelot-py opencv-python      # LIB-04 (also needs Ghostscript)
pip install tabula-py                     # LIB-05 (also needs JVM)
pip install python-docx                   # LIB-06
pip install python-pptx                   # LIB-07

# Tier 2 — AI-augmented
pip install docling                       # LIB-08
pip install mineru                        # LIB-09
pip install marker-pdf                    # LIB-10
pip install surya-ocr                     # LIB-11
pip install unstructured[all-docs]        # LIB-12
pip install markitdown                    # LIB-13

# Tier 3 — Specialised
pip install pytesseract                   # LIB-14 (also needs Tesseract binary)
pip install transformers timm             # LIB-15 (Table Transformer)
pip install nougat-ocr                    # LIB-16
pip install pdfminer.six                  # LIB-17
```

> **ARM64 Note:** Camelot (LIB-04) requires Ghostscript — verify ARM64 Ubuntu package availability (`apt install ghostscript`). Tabula-py (LIB-05) requires a JVM (`apt install default-jre`). All others install via pip without platform-specific dependencies.

---

## 6. Benchmark Datasets

### 6.1 Public Benchmarks (Tier 1)

| ID | Dataset | Focus | Priority |
|----|---------|-------|----------|
| D-01 | OmniDocBench | Overall — text, tables, math, layout (EN + ZH) | HIGH — primary benchmark |
| D-02 | FinTabNet | Financial table structure (annual reports) | HIGH — closest to automotive technical docs |
| D-03 | FUNSD | Form/field understanding on noisy scans | MEDIUM |
| D-04 | DocVQA | Document Q&A; form + table comprehension | MEDIUM |
| D-05 | OlmOCR-Bench | English precision; 1,400+ PDFs; **7,000+ binary unit tests** across 6 categories (math, tables, old scans, multi-column, tiny text, headers/footers) | **HIGH — primary unit-test evaluation framework** |
| D-06 | **DocLayNet** (IBM) | Real enterprise layout — financial, legal, scientific, government, patent documents; 80K+ pages; human-annotated bounding boxes for 11 element types | **HIGH — primary real-world enterprise benchmark** |

> **DocLayNet rationale:** DocLayNet provides real-world enterprise document diversity (multi-column, tables, headers, figures) that OmniDocBench lacks. Its human-annotated bounding boxes across 11 element types make it the strongest available public benchmark for evaluating extraction quality on enterprise-style documents.

> **Note:** This benchmark uses public datasets exclusively. No corporate or proprietary documents are included. The LLM-as-judge evaluation layer (L4) is applied to DocLayNet and OmniDocBench samples where ground truth annotations may be incomplete or ambiguous.

---

## 7. Functional Requirements

### 7.1 Inference & Deployment

**FR-01** Each model SHALL be deployable on the DGX Spark without external network calls during inference.

**FR-02** All models SHALL be served exclusively via `vllm serve` with an OpenAI-compatible endpoint on localhost.

**FR-03** Each model SHALL process input as page-level images (PNG/JPEG). PDF pre-conversion to images SHALL use `pdf2image` or equivalent.

**FR-04** The inference pipeline SHALL accept a directory of document images and produce structured output per page.

**FR-05** ARM64 compatibility SHALL be verified for each model and its dependencies before benchmarking begins.

### 7.2 Extraction Capabilities

**FR-06** Each model SHALL be evaluated on the following element types:

- Plain text (paragraphs, headings, footers)
- Tables (including multi-row, multi-column, merged cells)
- Mathematical expressions / equations
- Figures / charts (presence detection + caption extraction where supported)
- Reading order correctness across multi-column layouts

**FR-07** Where a model supports bounding box output, coordinates SHALL be normalised and transformed to pixel space for evaluation.

**FR-08** Table output SHALL be captured in the model's native format (LaTeX, HTML, or Markdown) and converted to a canonical HTML representation for TEDS scoring.

### 7.3 Output & Logging

**FR-09** Raw model outputs SHALL be saved per page per model in a structured directory:
```
results/
  {model_id}/
    {dataset_id}/
      {document_id}/
        page_{n}.json   # structured output
        page_{n}.txt    # plain text extraction
```

Library outputs SHALL follow an equivalent structure:
```
results/
  {lib_id}/
    {dataset_id}/
      {document_id}/
        page_{n}.txt      # extracted text
        page_{n}_tables.json  # table data (where applicable)
```

**FR-10** A benchmark summary report SHALL be generated in Markdown and CSV formats upon completion of all model and library runs.

### 7.4 Library-Specific Requirements

**FR-11** Each Tier 1 and Tier 2 library SHALL be tested on the same document corpus as the VLM models (Tier 1 public benchmarks) to enable direct comparison.

**FR-12** Libraries SHALL be evaluated on digital (non-scanned) documents independently of OCR-dependent models. Scanned document tests SHALL combine Tier 1 libraries with LIB-14 (Tesseract) or LIB-11 (Surya) as the OCR backend.

**FR-13** Table extraction results from LIB-02 (pdfplumber), LIB-04 (Camelot), LIB-05 (Tabula-py), and LIB-15 (TATR) SHALL each be converted to canonical HTML for TEDS scoring alongside VLM table outputs.

**FR-14** Library processing time SHALL be measured in isolation (no model inference) to establish the cost of the ingestion layer separately from the VLM layer.

### 7.5 Evaluation Methodology

**FR-15** All model and library text outputs SHALL be normalised to canonical Markdown format (per Section 9.3) before any metric evaluation. The normalisation script SHALL be deterministic and reproducible.

**FR-16** Each model SHALL be evaluated using all four metric layers defined in Section 9.1: (L1) CER/WER via `jiwer`, (L2) SBERT cosine similarity via `sentence-transformers`, (L3) binary unit tests via OlmOCR-Bench framework, and (L4) LLM-as-judge for samples where ground truth annotations are incomplete or ambiguous. All four scores SHALL be reported per model in the benchmark summary.

**FR-17** The LLM-as-judge evaluation (L4) SHALL use Qwen2.5-7B (L-01) served via vLLM on the DGX Spark with temperature=0 and fixed random seed. Judge prompts SHALL be versioned and stored in `eval/prompts/`. All judge inputs and responses SHALL be logged for reproducibility.

**FR-18** Throughput measurements SHALL exclude the first 3 pages per model per dataset as warm-up (see Section 9.4). Cold-start latency SHALL be logged separately.

---

## 8. Non-Functional Requirements

**NFR-01 Performance:** Each model SHALL complete single-page inference within 30 seconds on the DGX Spark at standard precision (bfloat16). Where quantisation is required for VRAM constraints, the quantisation method and level SHALL be logged alongside results.

**NFR-02 Memory:** Total GPU memory usage per model run SHALL be logged. Models exceeding available VRAM SHALL be flagged but not excluded (CPU offload or 4-bit quantisation may be attempted and documented).

**NFR-03 Reproducibility:** All inference runs SHALL use fixed random seeds and deterministic decoding (temperature=0, top_k=1 where applicable) unless stability issues require adjustment — any deviations SHALL be documented explicitly per model.

**NFR-04 Isolation:** Each model SHALL run in a separate Python virtual environment (`uv venv`) to prevent dependency conflicts.

**NFR-05 Logging:** Inference time per page, GPU memory peak, quantisation level (if applicable), and any generation errors SHALL be logged for every run.

**NFR-06 No Data Egress:** No document content, model outputs, or telemetry SHALL leave the DGX Spark during or after benchmarking. All HuggingFace model downloads SHALL be completed prior to the air-gapped benchmark phase.

**NFR-07 Library Isolation:** Each library benchmark SHALL run in its own `uv venv` to prevent version conflicts — particularly between libraries sharing pdfminer.six as a base (pdfplumber, Unstructured).

**NFR-08 Library Throughput Baseline:** Tier 1 libraries SHALL complete single-page extraction in under 1 second on the DGX Spark (CPU-only). Tier 2 libraries (GPU-optional) SHALL complete in under 15 seconds. These are advisory targets to flag unexpectedly slow installs.

---

## 9. Evaluation Metrics

### 9.1 Multi-Layer Evaluation Framework

Traditional CER/WER metrics alone are insufficient for VLM-based extraction — they penalise benign formatting differences, treat all errors equally (a misread character vs. a scrambled column), and cannot assess semantic correctness. This benchmark uses a **four-layer evaluation approach**, each layer catching error classes the others miss.

| Layer | Metric | What It Catches | Tool / Method |
|-------|--------|----------------|---------------|
| **L1 — Edit Distance** | CER, WER | Character/word-level transcription errors | `jiwer` |
| **L2 — Semantic Similarity** | SBERT cosine similarity | Meaning-preserving format variations that CER penalises | `sentence-transformers` (`MiniLM-L12-v2`) |
| **L3 — Binary Unit Tests** | Pass/fail assertions per page | Structural correctness: "does string X appear?", "is cell A above B?", "does equation render?" | OlmOCR-Bench framework (`pip install olmocr`) + custom domain-specific assertions |
| **L4 — LLM-as-Judge** | Semantic accuracy score (0–10) | Holistic extraction quality where ground truth is unavailable or ambiguous | Local Qwen2.5-7B (L-01) on DGX Spark |

### 9.2 Metric Details

**L1 — Edit Distance (CER/WER):**
Retained as the baseline quantitative metric for cross-model comparison and comparability with published benchmarks. Computed using `jiwer` after output normalisation (see Section 9.3).

**L2 — Semantic Similarity (SBERT):**
Sentence-BERT embeddings (`all-MiniLM-L12-v2`) compute cosine similarity between extracted and ground truth text blocks. Used alongside CER to detect cases where CER falsely penalises correct-but-reformatted output. A model scoring high SBERT but high CER indicates formatting differences, not extraction errors. Used by AIMultiple's DeltOCR Bench as the primary accuracy metric.

**L3 — Binary Unit Tests (OlmOCR-Bench):**
Deterministic pass/fail assertions per document page. OlmOCR-Bench provides 7,000+ pre-built tests across 6 categories. Custom assertions can be added targeting domain-specific terminology, table cell relationships, and unit/symbol extraction correctness. Test categories:
- Presence/absence of key phrases
- Table cell ordering and placement
- Formula rendering equivalence (via rendered image comparison)
- Reading order correctness across multi-column layouts
- N-gram non-repetition (catches VLM repetition loops)

**L4 — LLM-as-Judge:**
For benchmark samples where formal ground truth is incomplete or ambiguous (e.g., complex multi-column layouts, degraded scans), use Qwen2.5-7B (L-01, already in the benchmark stack) as a local semantic judge. The judge receives the source image + extracted text and scores accuracy on a 0–10 scale across dimensions: text completeness, table fidelity, reading order, and figure/caption handling. Recent research (Horn & Keuper, 2025) shows LLM-based evaluation achieves Pearson r=0.78 with human judgment for formula assessment — substantially higher than CDM (r=0.34) or text similarity (r≈0). Chandra's team demonstrated this approach raises effective scores from 83.1% to 93.9% when semantic evaluation replaces strict string matching.

> **Note:** L4 runs on the same DGX Spark hardware and uses the LLM backbone already deployed for L-01 evaluation — no additional infrastructure cost.

### 9.3 Output Normalisation Protocol

All model outputs SHALL be normalised to a canonical Markdown representation before any metric evaluation. This prevents measuring format differences instead of extraction quality.

Normalisation steps (applied in order):
1. Strip model-specific preamble/postamble (e.g., `<|im_start|>`, system tokens)
2. Normalise Unicode (NFC form) — resolves smart quote / ligature variants
3. Collapse multiple whitespace / blank lines to single
4. Standardise table format to pipe-delimited Markdown (`| cell | cell |`)
5. Normalise inline formulas to Unicode where possible (per OmniDocBench v1.5 convention)
6. Strip page headers/footers if not part of the evaluation target

### 9.4 Warm-Up Protocol

First inference on vLLM includes model loading and CUDA kernel compilation, which materially inflates per-page latency. To ensure throughput measurements reflect steady-state performance:
- **Warm-up:** First 3 pages per model per dataset are excluded from throughput timing
- **Measurement window:** Pages 4 onwards are timed and included in pages-per-minute calculation
- **Cold-start logging:** The warm-up pages' latency SHALL be logged separately (as `cold_start_latency_ms`) for infrastructure planning purposes

### 9.5 Element-Level Metrics

| Element | Metric | Tool / Method |
|---------|--------|---------------|
| Text accuracy | CER, WER (L1) + SBERT cosine similarity (L2) | `jiwer`, `sentence-transformers` |
| Table structure | TEDS score (Tree-Edit Distance) | `table-transformer-teds` or custom |
| Layout / Bbox | IoU (Intersection over Union) per element | Custom script / COCODet (mAP/mAR) via OmniDocBench |
| Reading order | Normalised Edit Distance over block sequences | OmniDocBench v1.5 evaluation code |
| Figure detection | Precision / Recall (presence detection only) | Manual |
| Formula accuracy | CDM (Character Detection Matching) + LLM-as-judge (L4) | OmniDocBench CDM integration + local Qwen2.5-7B |
| End-to-end RAG | Retrieval recall on known-answer Q&A pairs | Custom RAG eval on DocVQA (D-04) |
| Throughput — models | Pages per minute (at bfloat16 / quantised), excl. warm-up | Timed runs (see Section 9.4) |
| Throughput — libraries | Pages per minute (CPU / GPU) | Timed runs |
| Memory — models | Peak GPU VRAM (GB) | `nvidia-smi` / DCGM |
| Memory — libraries | Peak RAM (GB, CPU libraries) | `psutil` |
| Library vs model delta | CER + SBERT difference between library-only and library+VLM | Per-layer comparison |

---

## 10. Acceptance Criteria

| Criterion | Threshold |
|-----------|-----------|
| Text CER on OmniDocBench — models | ≤ 5% |
| Text CER on OmniDocBench — libraries (digital PDF) | ≤ 8% |
| SBERT cosine similarity on OmniDocBench — models | ≥ 0.90 |
| OlmOCR-Bench unit-test pass rate — models | ≥ 70% overall |
| LLM-as-judge score on ambiguous/complex samples | ≥ 7.0 / 10 average |
| Table TEDS score | ≥ 0.85 |
| Bbox IoU (where supported) | ≥ 0.80 |
| Inference time per page — models (excl. warm-up) | ≤ 30s |
| Processing time per page — Tier 1 libraries | ≤ 1s (CPU) |
| Processing time per page — Tier 2 libraries | ≤ 15s (GPU-optional) |
| ARM64 deployment | Must succeed without source recompilation |
| No external API calls during inference | Mandatory — zero tolerance |

---

## 11. Infrastructure Requirements

| Component | Requirement |
|-----------|-------------|
| Hardware | NVIDIA DGX Spark (ARM64, Ubuntu) |
| CUDA | Compatible with Ampere / Hopper microarchitecture |
| Python | 3.10 or 3.12 (per model requirement) |
| vLLM | v0.14.1+ (supports Nemotron Parse v1.1 and Qwen2.5-VL) |
| Package manager | `uv` (preferred) or `conda` per environment |
| PDF conversion | `pdf2image` + `poppler` |
| Java (JVM) | Required for LIB-05 (Tabula-py); `apt install default-jre` |
| Ghostscript | Required for LIB-04 (Camelot); `apt install ghostscript` |
| Tesseract binary | Required for LIB-14 (pytesseract); `apt install tesseract-ocr` |
| Evaluation — CER/WER | `jiwer` |
| Evaluation — SBERT | `sentence-transformers` (model: `all-MiniLM-L12-v2`) |
| Evaluation — Unit tests | `olmocr` (OlmOCR-Bench framework) |
| Evaluation — OmniDocBench | OmniDocBench v1.5 eval code (`git clone opendatalab/OmniDocBench`); requires `latexml` for LaTeX→HTML table conversion |
| Evaluation — LLM-as-judge | Qwen2.5-7B (L-01) via vLLM — shared with LLM backbone benchmark |
| Storage | Minimum 300GB free (250GB models incl. M-12/M-13 + 50GB library deps + outputs + benchmark datasets) |
| Monitoring | `nvidia-smi`, `nvtop`, `psutil` (for CPU library profiling), DCGM (if compatible) |

---

## 12. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ARM64 incompatibility for a model | Medium | High | Pre-check each model's CUDA kernels before full run; use Docker fallback |
| vLLM version conflicts across models | Medium | Medium | Isolate each model in its own `uv venv`; pin vLLM version per model |
| LightOnOCR-2 stability / repetition loops | High | Medium | Set temperature=0.2, repetition_penalty=1.1, top_k=40 as defaults; log all deviations |
| GPU OOM for 7B models (OlmOCR-2, Qwen2.5-VL-7B) | Medium | Medium | Attempt 4-bit quantisation via vLLM (`--quantization awq`); document result |
| GPU OOM for Qwen2.5-VL-72B (L-03) | High | Medium | Q4 quantisation mandatory (AWQ via vLLM); schedule last after all smaller models complete |
| GOT-OCR 2.0 ARM64 build issues | Medium | Medium | Test transformers-only path first; vLLM optional |
| Nemotron Parse v1.1 → v1.2 licence change | Low | Medium | Verify NVIDIA Open Model Licence compatibility before substituting v1.2 |
| GLM-OCR PP-DocLayoutV3 sub-dependency ARM64 install | Medium | Medium | Test install in isolation before scheduling M-06; both MIT + Apache 2.0 licences acceptable |
| Nemotron-3-Nano (30B) VRAM on DGX Spark | Medium | Medium | Use vLLM quantisation (`--quantization awq`); evaluate as LLM backbone only, not OCR frontend |
| **albumentations ARM64 compatibility (Nemotron Parse postprocessing)** | **Medium** | **HIGH — BLOCKER** | **Resolve before any other benchmark work — see Section 12.1 below** |
| Camelot (LIB-04) Ghostscript ARM64 availability | Medium | Medium | Test `apt install ghostscript` on DGX Spark before scheduling; fallback to pdfplumber |
| Tabula-py (LIB-05) JVM ARM64 stability | Low | Medium | Verify `default-jre` ARM64 package; alternative is camelot stream mode |
| MinerU (LIB-09) PDF-only constraint | Low | Low | Apply only to PDF corpus; use Docling for DOCX/PPTX layers |
| Unstructured (LIB-12) scanned PDF accuracy | Medium | Medium | Use with LIB-11 (Surya) or LIB-14 (Tesseract) as OCR backend; do not run standalone on scanned docs |
| pdfminer.six version conflicts (shared base for pdfplumber + Unstructured) | Medium | Medium | Pin pdfminer.six version per venv; do not share envs between LIB-02 and LIB-12 |
| Chandra 2 (M-12) OpenRAIL-M licence — personal use | Low | Low | Personal use and research are explicitly permitted; no licence issue for this benchmark |
| Chandra 2 (M-12) ~4B params VRAM on DGX Spark | Low | Low | Well within VRAM budget; vLLM at bfloat16 |
| `sentence-transformers` ARM64 compatibility | Low | Medium | `MiniLM-L12-v2` is pure PyTorch — should install cleanly; test in isolation before scheduling |
| OlmOCR-Bench framework ARM64 install | Medium | Medium | Test `pip install olmocr` in throwaway venv; may require building from source if wheels missing |
| LLM-as-judge consistency / reproducibility | Medium | Medium | Use temperature=0, fixed seed; log all judge prompts and responses; validate against 10-page human-scored calibration set |

### 12.1 Blocker Resolution: albumentations ARM64 Compatibility (OQ-02)

**Context:** Nemotron Parse v1.1 (M-01) — the primary OCR candidate — uses `albumentations` as a postprocessing dependency for image augmentation/transforms during inference preprocessing. `albumentations` depends on compiled C extensions (`opencv-python`, `imgaug`) that may lack pre-built ARM64 wheels.

**Why this is a Day 1 blocker:** If `albumentations` cannot install on DGX Spark, M-01 is unrunnable and the entire benchmark priority order shifts. This must be resolved before setting up any other model environment.

**Resolution steps (execute in order):**

```bash
# Step 1: Create isolated throwaway venv
uv venv /tmp/test-albumentations
source /tmp/test-albumentations/bin/activate

# Step 2: Test the core sub-dependency first (most common ARM64 failure point)
pip install opencv-python-headless
# If this fails → try: pip install opencv-contrib-python-headless
# If both fail → install system OpenCV: apt install python3-opencv

# Step 3: Attempt full albumentations install
pip install albumentations
# If this succeeds → OQ-02 resolved, proceed with M-01 setup

# Step 4: If Step 3 fails, check the specific error:
# - If imgaug fails → pip install imgaug is optional for inference;
#   try: pip install albumentations --no-deps && pip install opencv-python-headless numpy scipy
# - If scikit-image fails → pip install scikit-image (usually has ARM64 wheels)

# Step 5: Verify the install
python -c "import albumentations; print(albumentations.__version__)"

# Step 6: Clean up
deactivate
rm -rf /tmp/test-albumentations
```

**Fallback if all steps fail:** Skip `albumentations` entirely and write a minimal custom preprocessing script using PIL + OpenCV directly. Nemotron Parse's core model does not require augmentations at inference time — only the reference pipeline wraps it for convenience. The custom script need only handle image resizing, normalisation, and colour space conversion.

**Outcome logging:** Record the result (pass/fail, which sub-step resolved it, any workarounds applied) in `setup/M-01/albumentations_arm64_check.log` for reproducibility.

---

## 13. Deliverables

| ID | Deliverable | Format |
|----|-------------|--------|
| DEL-01 | ExtractMark requirements document | `requirements.md` |
| DEL-02 | Environment setup scripts per model (vLLM) | `setup/{model_id}/install.sh` |
| DEL-03 | Environment setup scripts per library | `setup/{lib_id}/install.sh` |
| DEL-04 | Inference runner script (vLLM) | `run_benchmark.py` |
| DEL-05 | Library extraction runner script | `run_library_benchmark.py` |
| DEL-06 | Output normalisation script (Section 9.3) | `eval/normalise_output.py` |
| DEL-07 | Multi-layer evaluation runner (CER + SBERT + unit tests) | `eval/run_evaluation.py` (wraps OmniDocBench + OlmOCR-Bench + SBERT) |
| DEL-08 | LLM-as-judge prompt templates + runner | `eval/llm_judge.py` + `eval/prompts/` |
| DEL-09 | Raw results per model/dataset | `results/models/` directory |
| DEL-10 | Raw results per library/dataset | `results/libraries/` directory |
| DEL-11 | Benchmark summary report (models + libraries, all 4 metric layers) | `report/benchmark_summary.md` + `.csv` |
| DEL-12 | Model recommendation memo | `report/recommendation.md` |

---

## 14. Resolved Questions

All questions have been resolved. This section is retained as a decision log.

| ID | Question | Resolution |
|----|----------|-----------|
| OQ-01 | Nemotron Parse v1.2 licence? | Personal use — no blocker. Verify HuggingFace licence text before substituting v1.1 |
| OQ-02 | `albumentations` ARM64? | Resolution procedure in Section 12.1. Execute Day 1 |
| OQ-03 | Add dots.ocr? | No — covered by M-12 (Chandra 2) and M-13 (PaddleOCR-VL) |
| OQ-04 | PII anonymisation standard? | N/A — public datasets only, no corporate documents |
| OQ-05 | VRAM for 72B at Q8? | Q4 mandatory (AWQ via vLLM, ~36GB). Schedule L-03 last |
| OQ-06 | GLM-OCR PP-DocLayoutV3 ARM64? | Test via `pip install paddlepaddle paddleocr` in throwaway venv. If fails, M-06 deprioritised (core model runs independently) |
| OQ-07 | Ollama scope? | Dropped from formal benchmark (v1.3). vLLM only |
| OQ-08 | Nougat scope? | Scientific PDF subset only — D-01 (academic pages) and D-03 (FUNSD) |
| OQ-09 | Table output format? | HTML — serves TEDS scoring and downstream LLM extraction |
| OQ-10 | Chandra 2 commercial licence? | N/A — personal use explicitly permitted under OpenRAIL-M |
| OQ-11 | SBERT model? | `all-MiniLM-L12-v2` (English-optimised). Public datasets are predominantly English |
| OQ-12 | Omni AI JSON extraction framework? | Deferred to future pipeline design phase |

> **No outstanding blockers.** Benchmark can proceed once ARM64 pre-checks (OQ-02, OQ-06) are executed on Day 1.

---

## 15. Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-04-03 | Initial draft — 5 models, vLLM path only |
| v1.1 | 2026-04-03 | Added Ollama runtime path; expanded to 11 OCR frontend models + 3 LLM backbone candidates; added GLM-OCR, Qwen2.5-VL variants, InternVL2.5, OCRFlux-3B, DeepSeek-OCR; clarified Nemotron Parse vs Nemotron-3-Nano distinction; added Ollama pull command reference |
| v1.2 | 2026-04-03 | Added Section 5 — Libraries Under Evaluation (17 libraries across 3 tiers); added extraction stack architecture diagram; added library × format matrix; added library FRs (FR-12 to FR-15); added NFR-07 and NFR-08; expanded metrics, acceptance criteria, infrastructure, risks, deliverables, and open questions to cover library evaluation |
| v1.3 | 2026-04-03 | **Dropped Ollama** from formal benchmark scope — vLLM is sole benchmarking runtime (resolves OQ-07); replaced Ollama pull commands with vLLM serve reference; removed Ollama column from model tables; removed runtime delta metric and Ollama vs vLLM acceptance criterion. **Added DocLayNet** (D-06) as HIGH-priority public enterprise benchmark — replaces manual PHINIA annotation as primary real-world comparison; PHINIA samples downgraded to qualitative spot-check (D-09/10/11). **Added Section 12.1** — detailed albumentations ARM64 resolution procedure with step-by-step commands and fallback strategy (resolves OQ-02). |
| v1.4 | 2026-04-03 | **Multi-layer evaluation framework** — replaced single-metric CER/WER with 4-layer approach: (L1) CER/WER, (L2) SBERT semantic similarity, (L3) OlmOCR-Bench binary unit tests, (L4) LLM-as-judge via local Qwen2.5-7B. Added Section 9.1–9.5 covering framework design, metric details, output normalisation protocol, warm-up protocol, and element-level metrics. **New models** — added Chandra 2 (M-12, 4B, SOTA 85.9% olmOCR-Bench, OpenRAIL-M licence caveat) and PaddleOCR-VL (M-13, 0.9B, lightweight baseline). **Upgraded D-05** (OlmOCR-Bench) from LOW to HIGH priority — provides binary unit-test evaluation framework. **New FRs** — FR-16 (output normalisation), FR-17 (multi-layer evaluation), FR-18 (LLM-as-judge protocol), FR-19 (warm-up exclusion). **New acceptance criteria** — SBERT ≥ 0.90, OlmOCR-Bench pass rate ≥ 70%, LLM-as-judge ≥ 7.0/10. **Updated deliverables** — added DEL-06 (normalisation script), DEL-07 (multi-layer eval runner), DEL-08 (LLM-as-judge). **Added OQ-10** (Chandra 2 commercial licence), **OQ-11** (SBERT model selection), **OQ-12** (Omni AI JSON extraction framework). Added G-07, updated scope, infrastructure (evaluation deps, 300GB storage), and 6 new risks. |
| v1.5 | 2026-04-03 | **Scoped to personal use** — removed all PHINIA corporate document references; benchmark uses public datasets exclusively. **Removed D-09/10/11** (PHINIA spot-check samples) — no corporate documents in scope. **Resolved all 12 open questions:** OQ-03 (dots.ocr — No), OQ-04 (PII — N/A), OQ-05 (72B quantisation — Q4 mandatory), OQ-06 (PP-DocLayoutV3 — test procedure added), OQ-08 (Nougat — scientific subset only), OQ-09 (table format — HTML), OQ-10 (Chandra licence — personal use permitted), OQ-11 (SBERT — `all-MiniLM-L12-v2`), OQ-12 (Omni AI framework — deferred). **Updated M-12** (Chandra 2) — removed licence warning, personal use confirmed. **Updated L-03** — Q4 quantisation mandatory. Softened PHINIA references throughout library tables and scope sections. Zero open questions remain. |

---

*Project: ExtractMark | Owner: Kashif Muhammad | Personal AI Research*
*Status: All open questions resolved. Ready for execution — begin with ARM64 pre-checks (Section 12.1 albumentations, OQ-06 PaddlePaddle) on Day 1.*
