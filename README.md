# SANR-Embed
## *A Benchmark for Historical Legal Understanding & Cross-Lingual Bias*

### Abstract
**SANR-Embed** is a benchmark designed to evaluate the capabilities of Large Language Models (LLMs) in understanding, translating, and embedding 17th-century Spanish American legal texts. By utilizing a dataset of over 1,300 labeled records from the *Spanish Notary Collection* (1653–1658), this benchmark quantifies legal domain understanding, cross-lingual translation bias, and robustness to OCR noise. It addresses the critical gap in historical Spanish representation in modern AI research, offering a standardized framework to measure how Anglocentric bias affects the processing of archaic legal documents.

---

## Table of Contents
1. [Purpose](#1-purpose)
2. [Motivation](#motivation-why-this-benchmark-matters)
3. [Dataset (SANRLite)](#2-dataset-sanrlite)
4. [Tasks](#3-tasks)
5. [Model Classes](#4-model-classes)
6. [Evaluation Pipeline](#5-evaluation-pipeline)
7. [Testing Methodology](#6-testing-methodology-solid-principles)
8. [Directory Structure](#7-directory-structure)
9. [Conclusion](#conclusion)

---

## 1. Purpose

SANR-Embed evaluates how Language Models represent and reason over 17th-century Spanish American legal texts. It measures:

1. **Legal domain understanding** (Classification F1)
2. **Cross-lingual translation bias** (Performance Delta)
3. **OCR robustness** (Impact of visual noise)
4. **Embedding drift** (Vector space consistency)
5. **Temporal classification** (Diachronic shift)

SANR-Embed also serves as a benchmark for translation-conditioned embeddings. Because each model generates its own translations, the benchmark evaluates how translation style, lexical choice, and sentence restructuring affect the downstream embedding geometry and classification accuracy. This allows us to measure cross-lingual bias as a property of the entire translation → embedding → classification pipeline, rather than isolating only the embedding model.

---

## Motivation 

### **1. The theoretical AGI or a theoretically optimal LLM *should* be able to understand archaic legal texts — but we currently have no way to measure that.**  
General-purpose LLMs should be capable of reading, classifying, and reasoning over structured historical legal text. Archaic manuscripts, especially 17th-century notarial acts, are formulaic, patterned, and structurally stable — exactly the type of material a robust model should master.  
But until now, no benchmark has existed to test whether models genuinely understand archaic Spanish legal language, or whether they simply hallucinate structure. SANR-Embed establishes the missing baseline and grounds claims about “advanced reasoning” in measurable, historical-domain performance.

---

### **2. Spanish remains dramatically underrepresented in major pretraining corpora.**

Major LLM corpora overwhelmingly favor English and a handful of high-resource languages. Spanish — despite its global prevalence — appears far less often, which systematically weakens cross-lingual reasoning. SANR-Embed exposes how this imbalance affects classification, translation, and embedding behavior.


#### **1. The Pile — almost entirely English**  
One of the most widely used open pretraining corpora; Spanish presence is negligible.  
[Gao et al., 2020](https://arxiv.org/abs/2101.00027)

#### **2. C4 — 90%+ English; non-English filtered out**  
C4, used for T5 and many derivatives, intentionally removes most non-English text.  
[Raffel et al., 2020](https://arxiv.org/abs/1910.10683)

#### **3. Meta’s audit of multilingual corpora — high imbalance**  
Shows pretraining data is dominated by ~15 high-resource languages, with Spanish underrepresented relative to speaker base.  
[Nozza et al., 2023](https://arxiv.org/abs/2305.13168)

#### **4. Survey on Multilingual LLMs — English dominance persists**  
Confirms multilingual corpora remain heavily skewed toward English.  
[MLLM Survey, 2024](https://www.researchgate.net/publication/390498080_A_survey_on_multilingual_large_language_models_corpora_alignment_and_bias)

#### **5. Language Ranker — performance tied to data volume**  
Demonstrates LLM accuracy correlates strongly with training-data availability; under-resourced languages score worse.  
[Hickey et al., 2024](https://arxiv.org/abs/2404.11553)


---

### **3. The Spanish-speaking NLP community lacks accessible benchmarks.**  
Most evaluation frameworks assume English as the default language. Researchers working in Spanish, and especially those studying Latin American history, lack standardized tools to measure cross-lingual bias, OCR robustness, or legal-domain understanding.  
SANR-Embed offers an **easy-to-run, culturally grounded benchmark** that centers Spanish, gives researchers control over evaluation, and lowers the barrier to historical-domain NLP research.

---

### **4. It expands high-quality Spanish training data *without copyright risk*.**  
Modern Spanish corpora — news, books, academic databases, legal texts — are often paywalled or copyrighted. But 17th-century notarial manuscripts are fully in the **public domain**.  
This makes SANR-Embed not just an evaluation tool, but a safe source of:

- high-quality Spanish legal text  
- structured, supervised labels  
- morphologically rich examples  
- domain-specific reasoning signals  

Model developers can improve Spanish performance without licensing negotiations or scraping copyrighted sources.

---

## 2. Dataset (SANRLite)

**Source:** Public-domain historical manuscripts from the
**Spanish Notary Collection (1653–1658)**
https://github.com/raopr/SpanishNotaryCollection

Paper:
Shraboni Sarker, Ahmad Tamim Hamad, Hulayyil Alshammari, Viviana Grieco, and Praveen Rao. Seventeenth-Century Spanish American Notary Records for Fine-Tuning Spanish Large Language Models. In Proc. of 2024 ACM/IEEE-CS Joint Conference on Digital Libraries (JCDL 2024), 5 pages, 2024.

* **Period:** 1653–1658
* **Region:** Buenos Aires, Argentina
* **Domain:** Notarial law (Wills, Powers of Attorney, Sales, Labor Contracts)
* **Size:** ~1,300+ labeled records



## 3. Tasks

### A. Native Legal Classification
- **Input:** `text_original`
- **Target:** `label_primary`
- **Metric:** Macro F1
- Tests pure model capability on archaic Spanish.

### B. Cross-Lingual Bias (Δ-F1)
1. Translate text to English/Chinese.
2. Classify translated text.
3. Compute Δ using:
   \[
     \Delta = F1_{\text{translated}} - F1_{\text{native}}
   \]

Interpretation:
- **Δ > 0** → Anglocentric bias
- **Δ < 0** → Robust Spanish-native reasoning

**Protocol Note:** Ideally, each model should translate using its own translation capability (“self-translation”). Default CSVs (Google Translate) are also provided.

### C. OCR Robustness
Compare F1 between:
- Clean text (`text_original`)
- Noisy OCR (`ocr_noisy`)

Metric: performance drop Δ_OCR.

### D. Embedding Drift
Compute cosine similarity between:
- Spanish → English → Spanish
- Spanish vs. English parallel sentences

### E. Temporal Classification (Optional)
Predict year class (e.g., 1653 vs 1658) to measure diachronic linguistic sensitivity.

---

## 4. Model Classes

### Tier 1 — Baselines
- Logistic Regression (TF-IDF, BERT embeddings)
- BETO (Spanish BERT)
- mBERT (Multilingual)

### Tier 2 — Open Weights
- DeepSeek-V3
- Llama-3
- Qwen-2.5

### Tier 3 — Proprietary
- GPT-4o
- Claude 3.5 Sonnet

### 4.5 Model Adapter Interface

```python
class ModelAdapter:
    def translate(self, text: str, target_lang: str) -> str: ...
    def embed(self, text: str) -> np.ndarray: ...
    def classify(self, text: str, label_set: list[str]) -> str: ...
    
    # Optional:
    def score_batch(self, texts): ...
    def fine_tune(self, ...): ...
    def reset(self): ...
```

### 4.6 Registering a New Model

Add the model in src/_init_.py


### Registering the fine tuned model
https://github.com/raopr/SpanishNotaryCollection
Download, place inside the `src\models\` inside respsective classifcation or masked_language_model folders.
The `SANR-Embed\src\models\classification_adapter.py` and `SANR-Embed\src\models\masked_lm_adapter.py` are made to adapt it to the registry. I would incorporate it in the repo but they're too loarge. 

### Registering 

---

## 5. Evaluation Pipeline

### 5.1 Fine-Tuning (Optional)
- K-Fold CV (default k=5)
- Automatic fallback for minority classes
- Ensures model independence via state resets

### 5.2 Standard Pipeline
`main.py` performs:
1. Load model + dataset
2. Fine-tuning (optional)
3. Translation
4. Classification
5. Embedding
6. Reporting metrics (F1, Δ, drift)

---

## 6. Testing Methodology (SOLID Principles)

### 6.1 Strategy
- Unit tests per module
- Integration tests with mock models
- Mocking avoids API calls during CI

### 6.2 SOLID Principles
- **SRP:** Each test checks one behavior
- **OCP:** New models don’t require test rewrites
- **LSP:** `MockModelAdapter` substitutes real adapters
- **ISP:** Tests only use relevant interface methods
- **DIP:** Tests depend on abstractions, not implementations

Run Tests:
```bash
pytest tests/
```

---

## 7. Directory Structure

```
SANR-Embed/
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── gold_standard.csv
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── embeddings.csv
│   ├── images/
├── src/
│   ├── api/
│   ├── processing/
│   ├── embeddings/
│   ├── classifiers/
│   ├── models/
│   └── analysis/
├── tests/
├── results/
├── requirements.txt
└── DESIGN.md
```

---

## Conclusion

SANR-Embed provides the first systematic benchmark for evaluating how modern LLMs understand, translate, classify, and embed historical Spanish American legal texts. It exposes structural weaknesses in multilingual models, highlights English-centric biases, and offers the Spanish-speaking research community a practical, copyright-safe tool for improving LLM performance. By grounding evaluation in real colonial manuscripts, SANR-Embed pushes the field toward fairer, more historically-aware, and more linguistically inclusive AI systems.

**Author Website:** https://mjrobillard.com
