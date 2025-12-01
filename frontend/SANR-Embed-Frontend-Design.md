# SANR-Embed Frontend Design Document

## 1. Overview
The SANR-Embed Frontend is a web-based interface designed to democratize access to the SANR-Embed benchmark. It serves three primary goals:
1.  **Education:** Explain the project's purpose (evaluating historical legal understanding and cross-lingual bias) to a broader audience.
2.  **Benchmarking:** Provide an interactive dashboard to visualize model performance, compare biases, and potentially trigger new evaluations.
3.  **Digital Humanities Showcase:** Act as a gallery for the *SANRLite* dataset, allowing historians and researchers to explore the 17th-century manuscripts alongside their transcriptions and metadata.

---

## 2. Technology Stack

### 2.1 Frontend Framework
*   **Framework:** **Next.js** (React)
    *   *Reasoning:* Excellent for static generation (fast landing pages), server-side rendering (SEO for the dataset showcase), and easy API integration.
*   **Language:** **TypeScript**
    *   *Reasoning:* Type safety for the complex data structures (benchmark metrics, document metadata).
*   **Styling:** **Tailwind CSS**
    *   *Reasoning:* Rapid development, consistent design system.
*   **Visualization:** **Recharts** or **Nivo**
    *   *Reasoning:* High-performance charting for F1 scores, embedding drifts, and bias deltas.
*   **Image Viewing:** **React-Zoom-Pan-Pinch**
    *   *Reasoning:* Essential for examining detailed handwriting in historical manuscripts.

### 2.2 Backend / API Layer
Since the core benchmark is Python-based, we require a lightweight API layer to bridge the CLI tools with the frontend.
*   **Framework:** **FastAPI** (Python)
    *   *Reasoning:* Native integration with the existing `src/` Python codebase. Async support for handling long-running benchmark jobs.
*   **Database:** **SQLite** (or direct CSV reading)
    *   *Reasoning:* The dataset (~1,300 records) is small enough to be served directly from the `gold_standard.csv` or a lightweight SQLite db.

---

## 3. User Interface & Architecture

### 3.1 Site Map
1.  **Home (/)**: Project introduction, methodology visualization, high-level summary metrics.
2.  **Leaderboard (/benchmark)**: Interactive table and charts comparing models (Tier 1, 2, 3).
3.  **Dataset Explorer (/data)**: Searchable gallery of documents.
    *   **Document Detail (/data/[id])**: Split-screen view (Image vs. Text).
4.  **Embedding Explorer (/embeddings)**: Interactive 2D/3D visualization of the vector space, colored by metadata.
5.  **Run Benchmark (/run)**: Interface to configure and trigger evaluations (optional/advanced).
6.  **About (/about)**: Team, citation info, documentation.

### 3.2 Detailed Page Designs

#### A. Home Page
*   **Hero Section:** Title, subtitle, and a background visual of a manuscript.
*   **"What is SANR-Embed?"**: Brief explanation of the challenges (Archaic Spanish, Legal Domain, OCR Noise).
*   **Key Insight:** A dynamic "Bias vs. Accuracy" scatter plot (immediately visible).

#### B. Benchmark Leaderboard
A rich dashboard for analyzing model performance.
*   **Controls:**
    *   Toggle Tasks: "Native Classification", "Cross-Lingual Bias", "OCR Robustness".
    *   Filter Models: "Open Weights", "Proprietary", "Baselines".
*   **Visualizations:**
    *   **The Bias Plot:** X-axis = Native F1, Y-axis = Cross-Lingual Delta ($\Delta$).
        *   *Goal:* Identify models in the "Sweet Spot" (High Accuracy, Low Bias).
    *   **Bar Charts:** Side-by-side comparison of specific metrics.
*   **Table:** Sortable columns for F1, $\Delta$, OCR Drop, Inference Cost/Speed.

#### C. Dataset Showcase (The "Digital Archive")
Designed for historians and linguists.
*   **Gallery Grid:** Cards showing a thumbnail of the manuscript, the Year, Notary, and Document Type (e.g., "Poder especial").
*   **Filters:**
    *   **Year Range:** 1653–1658.
    *   **Notary:** Filter by specific author.
    *   **Legal Class:** "Venta", "Testamento", etc.
*   **Document Detail View (Split Screen):**
    *   **Left Panel (Image):** High-res Deep Zoom viewer for the original scan.
    *   **Right Panel (Data):**
        *   **Tabs:** "Transcription (Original)", "Translation (English/Chinese)", "Metadata".
        *   **Metadata Box:** Rollo number, Image number, Wikidata link.
        *   **Model Analysis (Optional):** Show how different models classified this specific document.

#### D. Embedding Space Explorer
A powerful tool to visualize how different models represent the 17th-century legal landscape.
*   **Interactive Scatter Plot (UMAP/t-SNE):**
    *   Each dot represents a document record.
    *   **Model Selector:** Switch between "Native Spanish", "English Translation", or different model architectures (e.g., mBERT vs. DeepSeek).
*   **Color/Group By:**
    *   **Legal Class (`label_primary`):** See if wills, sales, and contracts form distinct clusters.
    *   **Notary (`notary`):** Detect if specific authors have unique stylistic fingerprints.
    *   **Year (`year`):** visualize temporal drift (1653 vs. 1658).
    *   **Rollo (`rollo`):** Check for archival bundle artifacts.
*   **Hover Tooltips:** Show the snippet of text and the image thumbnail on hover.
*   **Search/Highlight:** Highlight specific terms or IDs within the cloud.

#### E. Run Interface (Advanced)
*   **Configuration Form:**
    *   Select Model Adapter (dropdown from `MODEL_REGISTRY`).
    *   Select Tasks (A, B, C, D, E).
    *   Upload/Select Data Split.
*   **Status Console:** Real-time log stream from the backend execution (WebSocket).
*   **Results Preview:** JSON summary of the run once complete.

---

## 4. Data Flow & API Endpoints

### 4.1 Static Data (Read-Only)
The frontend primarily consumes the results generated by the benchmark.
*   `GET /api/results`: Returns parsed JSON from `results/*.csv` (aggregated metrics).
*   `GET /api/documents`: Returns paginated list of documents from `gold_standard.csv`.
*   `GET /api/documents/{id}`: Returns details for a single document.
*   `GET /images/{image_path}`: Serves the static image files.

### 4.2 Interactive Data (Write/Execute)
*   `POST /api/benchmark/run`: Triggers a background task (via Celery or generic `asyncio` task) to run `main.py`.
    *   *Payload:* `{ model_name: "deepseek", tasks: ["A", "B"] }`

---

## 5. Implementation Plan

### Phase 1: Showcase & Static Results
1.  **Setup Next.js**: Scaffold project.
2.  **Data Loader**: Script to convert `gold_standard.csv` into a JSON index for the frontend.
3.  **Gallery Component**: Build the masonry grid for documents.
4.  **Viewer Component**: Implement zoomable image viewer.
5.  **Leaderboard**: Hardcode the initial results from `results/` into a Recharts visualization.

### Phase 2: Backend Integration
1.  **FastAPI Setup**: Wrap `src/` in a Python API.
2.  **Live Results**: Fetch metrics dynamically from the backend.

### Phase 3: Interactive Benchmarking
1.  **Job Queue**: Implement a simple queue for running models (preventing server overload).
2.  **Run UI**: Build the form and status terminal.

