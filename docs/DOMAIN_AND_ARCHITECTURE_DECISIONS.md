# TinyRAG: Domain & Architectural Decisions

**Document Created:** October 11, 2026  
**Status:** Approved for Phase 7 (Capstone Exhibition & Final Defense)  
**Parent Repository:** `tinyrag` (Group 14, CSE 600, University of Chittagong)

---

## 1. Selected Application Domain: Air-Gapped Critical Infrastructure & Industrial Edge Assistant

### 1.1 Motivation & The "Why RAG on Edge?" Defense
* **Regulatory Air-Gap:** In critical industrial infrastructure (SCADA networks, power substations, high-voltage transformers, heavy manufacturing plants), public cloud internet access is strictly prohibited by law and cybersecurity regulations to prevent cyber espionage, data leakage, and remote sabotage.
* **Zero-Hallucination Grounding:** Operating and repairing complex machinery requires 100% adherence to verified manufacturer manuals. Unlike raw LLMs which can fabricate critical electrical pinouts or safety procedures, TinyRAG strictly grounds all responses in official documentation with exact page citations.
* **On-Device Telemetry + Documentation Synthesis:** Real-time sensor telemetry (ambient temperature, humidity, equipment thermal readings, vibrations) is processed locally on the edge device alongside tens of thousands of pages of technical operation and troubleshooting manuals.

### 1.2 Alternative / Extensible Domains Preserved
The core architecture remains extensible. The following domains can be targeted by simply ingesting domain-specific corpora:
1. **AgriRAG (Off-Grid Agriculture):** Remote farmer assistant for crop diseases and soil moisture guidelines where rural internet is intermittent.
2. **Maritime & Off-Grid Mining:** Deep-sea vessels, offshore platforms, and underground mining shafts where satellite links are cost-prohibitive and high-latency.
3. **Medical & Laboratory Diagnostic OEM:** Embedded on-premise assistant for sensitive medical diagnostic machines adhering to strict privacy laws (HIPAA/GDPR).

---

## 2. Technical Manual Image & Diagram Handling Strategy

Technical manuals are heavily visual (wiring diagrams, schematics, component pinouts, fault charts). To handle these on resource-constrained edge hardware (e.g., Raspberry Pi 5 with 8 GB RAM) without causing thermal or memory overload:

### 2.1 Layer 1: Smart Figure Anchoring & UI Projection (Zero Extra Compute)
* **Ingestion:** During document parsing, figures, captions, diagrams, and their bounding box coordinates on specific pages are indexed alongside adjacent text chunks.
* **Retrieval & Generation:** When a user queries a component or wiring step, the text RAG pipeline generates the step-by-step instructions and references the figure:
  > *"Connect the bypass relay terminal 4 to ground. **[Refer to Wiring Diagram 3.2 on Page 24]**"*
* **Web UI Rendering:** The Web UI automatically retrieves the extracted figure crop from that page and displays the high-resolution diagram adjacent to the answer. The human engineer sees the visual schematic immediately, while the Pi 5's LLM performs zero visual inference.

### 2.2 Layer 2: Ingestion-Time OCR & Table Extraction
* Diagrams with embedded text (pinout tables, error code charts) are processed during the ingestion phase using lightweight OCR (Tesseract / table parsers). This text is indexed into FAISS, making all diagram annotations fully searchable.

### 2.3 Layer 3: Future Roadmap (Tiny Edge VLMs)
* Small Vision-Language Models (e.g., Moondream2 1.8B, SmolVLM, or Gemma multimodal) can be plugged in as modular add-ons when hardware acceleration or external camera visual inspections are needed.

---

## 3. Git Branching & Versioning Strategy

To keep the project clean, modular, and flexible:
* **`main` Branch:** Remains **Domain-Agnostic and General-Purpose**. Contains the verified core RAG pipeline, test suite (1400+ tests), and modular interfaces.
* **`domain/airgapped-industrial` Branch:** Dedicated feature branch for industrial manuals, sample SCADA telemetry datasets, domain-specific UI themes, and demonstration artifacts.
* Allows instantaneous switching (`git checkout main`) for evaluation against any other general corpus.

---

## 4. Immediate Milestone: Poster Presentation for Department Exhibition
* **Format:** Standard academic conference exhibition poster (36" × 48", Portrait, 3-Column Layout).
* **Target Audience:** Department faculty, evaluators, industry visitors, and peer students at CSE, University of Chittagong.
* **Next Steps:** Layout planning, visual generation, content drafting, and vector rendering.
