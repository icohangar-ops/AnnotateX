# BUIDL: AnnotateX

## BUIDL Information
- **Project Name**: AnnotateX
- **Vision**: Long-context ICL-powered automatic data annotation engine for next-gen AI training pipelines. (88 chars)
- **Category**: AI Agent / LLM Application
- **Is this BUIDL an AI Agent?**: Yes — AnnotateX is an autonomous ICL-based annotation agent that reads long-context documents, reasons through annotation decisions using chain-of-thought, and validates outputs via self-consistency decoding.
- **Logo**: https://github.com/icohangar-ops/AnnotateX/blob/main/logo.png
- **GitHub Repo**: https://github.com/icohangar-ops/AnnotateX

---

## BUIDL Description (< 30,000 characters)

### What is AnnotateX?

AnnotateX is an intelligent, open-source data annotation engine designed to solve one of the most critical bottlenecks in modern AI development: **scalable, high-quality data annotation in long-context scenarios**. Built for the FlagOS Open Computing Global Challenge (Track 3), AnnotateX leverages In-Context Learning (ICL) with Qwen3-4B to automatically annotate complex datasets — reducing what traditionally takes human teams weeks into minutes of GPU compute time.

### The Problem

As large language models push beyond 32K context windows, the demand for high-quality annotated training data has exploded. Traditional annotation workflows rely heavily on human annotators who must read, understand, and label documents that span tens of thousands of tokens. This process is:
- **Expensive**: Professional annotation costs $0.10–$2.00 per label
- **Slow**: Long-context documents take 15–45 minutes per annotation
- **Inconsistent**: Inter-annotator agreement drops below 70% for complex tasks
- **Non-scalable**: Cannot keep pace with the data hunger of modern LLMs

### Our Solution

AnnotateX addresses these challenges through a multi-layered ICL architecture:

1. **Few-Shot Selection**: Selects few-shot examples per task — binary classification tasks use balanced per-label sampling so both labels are represented; all other tasks use uniform random sampling. *(Reworded from the original claim of "embedding-based similarity clustering", which named no code path in this repository.)*

2. **Answer-Only ICL Prompting**: Each prompt carries the task definition and numbered few-shot examples, and the system message requests only the final answer. *(Reworded from the original claim of "Chain-of-Thought reasoning: forces the model to reason step-by-step"; the committed prompt requests answer-only output and contains no CoT scaffold.)*

3. **Self-Consistency Decoding (optional)**: When `SELF_CONSISTENCY_RUNS` is set above 1, the engine runs multiple inference passes with varied temperature settings and aggregates results via majority voting. The committed default is `SELF_CONSISTENCY_RUNS = 1` (single pass, no voting).

4. **Long-Context Truncation**: Inputs are truncated by the tokenizer at `MAX_INPUT_TOKENS = 30000`, inside Qwen3-4B's 32K token window. *(Reworded from the original claim of "intelligent context truncation and document chunking strategies that preserve the most semantically relevant portions"; no chunking or relevance-based selection is implemented.)*

5. **Task-Type-Adaptive Handling**: Prompt construction and answer extraction adapt to the task's label schema — binary and multi-class classification get label-balanced selection and known-label matching; all other tasks fall back to uniform sampling and generic extraction. *(Reworded from the original claim of "structured entity extraction" handling, which named no code path in this repository.)*

### Technical Architecture

```
┌─────────────────────────────────────────────────────┐
│                  AnnotateX Pipeline                  │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │ Raw Data  │──▶│  ICL Example │──▶│  Prompt      │ │
│  │ (CSV/JSON)│   │  Selector    │   │  Builder     │ │
│  └──────────┘   └──────────────┘   └──────┬───────┘ │
│                                             │         │
│                                             ▼         │
│  ┌──────────────────────────────────────────────────┐ │
│  │            Qwen3-4B (4-bit Quantized)             │ │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────────────┐  │ │
│  │  │ CoT     │  │ Multi-   │  │ Self-Consistency │ │ │
│  │  │ Reasoning│  │ Temp Run │  │ Majority Vote   │ │ │
│  │  └─────────┘  └──────────┘  └────────┬────────┘  │ │
│  └───────────────────────────────────────┼──────────┘ │
│                                          │            │
│                                          ▼            │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Label        │──▶│ Answer       │──▶│ Submission│ │
│  │ Extractor    │    │ Extractor    │   │ (CSV)     │ │
│  └───────────────────────────────────────┴───────────┘ │
│                                                       │
└─────────────────────────────────────────────────────┘
```

### Reliability & Resilience

Generation is wrapped with `cubiczan_resilience`'s `@resilient(timeout=300, max_attempts=3)`, so transient inference failures are retried with exponential backoff. Each sample is additionally retried up to `MAX_PREDICT_ATTEMPTS = 3`; a sample that exhausts all attempts falls back to an empty prediction and is recorded in an auditable error sidecar (`<submission>.errors.json`) written atomically via `cubiczan_resilience.atomic_write`. GPU memory is cleared between failed attempts and periodically during long runs.

### Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Base Model | Qwen3-4B (FlagOS variant) | Core inference engine |
| Runtime | PyTorch + Transformers | Model execution and tokenization |
| Quantization | bitsandbytes (NF4 4-bit) | Memory-efficient deployment (requested via `USE_4BIT`) |
| Prompting | Few-shot ICL, answer-only outputs | Format-faithful labels per task definition |
| Validation | Self-consistency decoding | Majority-vote aggregation when `SELF_CONSISTENCY_RUNS > 1` |
| Context | Tokenizer truncation | 30,000-token input window (`MAX_INPUT_TOKENS`) |

### Performance Characteristics

- **Inference Speed**: ~3-8 seconds per annotation in the competition run (GPU-dependent; observation from that run, not re-verified in CI)
- **Memory Usage**: ~8-10 GB VRAM in the competition run with 4-bit quantization (observation from that run, not re-verified in CI)
- **Context Handling**: inputs truncated at 30,000 tokens (`MAX_INPUT_TOKENS`) within Qwen3-4B's 32K token window
- **Accuracy**: no accuracy figure is re-verified in CI; self-consistency with 3 runs showed 85-95% agreement during the competition run (the committed default is 1 run)
- **Scalability**: inference is sequential per sample (no batched generation); `device_map="auto"` with fp16 weights lets the model shard across available GPUs

### Open Source & Community

AnnotateX is fully open-source. All code and the Kaggle competition notebook are available in this repository (there are no separate evaluation scripts — prompt construction lives in `icl_annotation_solver.py`). We actively contribute to the FlagOS ecosystem and OpenSeek repository. A technical report is committed at `Technical_Report.pdf` and is regenerable with `generate_report.py`.

### Future Roadmap

- **FlagScale Integration**: Native deployment on FlagScale for multi-chip distributed inference
- **Multi-Model Support**: Extension to other models in the Qwen family and beyond
- **Active Learning Loop**: Feedback mechanism to iteratively improve annotation quality
- **Enterprise API**: RESTful API for production annotation pipelines
- **Domain Specialization**: Pre-configured annotation templates for medical, legal, and financial domains

---

## GitHub Repository

**Repository URL**: https://github.com/icohangar-ops/AnnotateX

### Repository Structure:
```
AnnotateX/
├── icl_annotation_solver.py         # Core ICL annotation engine
├── generate_report.py               # Technical report (PDF) generator
├── kaggle_notebook.ipynb            # Kaggle competition notebook
├── tests/test_engine.py             # CPU-only engine tests (evidence matrix)
├── scripts/                         # Deterministic claim-verification scripts
├── evidence/matrix.yaml             # Capability claims bound to evidence
├── tools/verify_evidence_matrix.py  # Vendored fail-closed verifier
├── assets/demo.mp4                  # Short demo clip
├── demo-video.mp4                   # Demo video
├── logo.png                         # Project logo
├── Technical_Report.pdf             # Committed technical report
└── README.md                        # Documentation
```

---

## Demo Video

The demo video is committed in this repository at [demo-video.mp4](demo-video.mp4) — a short clip is also available at [assets/demo.mp4](assets/demo.mp4). *(Reworded from the original external link, which pointed at a different project's media.)*

---

## Built With

- **Qwen3-4B** by Alibaba (via FlagOS release) — core inference engine
- **PyTorch** — model runtime
- **Transformers** by HuggingFace — model/tokenizer loading and chat templating
- **numpy / pandas** — data handling
- **cubiczan-resilience** — retry/backoff (`resilient`) and atomic file writes (`atomic_write`)
- **Kaggle** — competition platform (see `kaggle_notebook.ipynb`)

*(Reworded from the original list, which named FlagScale and FlagGems although neither appears in the requirements or the code; FlagScale remains a roadmap item.)*

---

## Evidence matrix

Every capability claim in this file is backed by
[`evidence/matrix.yaml`](evidence/matrix.yaml); CI refuses builds while any row
is unverifiable (run `python3 tools/verify_evidence_matrix.py` locally).

---

## Team

**zan-maker** — Solo developer and AI systems engineer
