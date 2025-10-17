# Web Crawling & Legal Text Analysis — Areios Pagos Decisions

### **Project Overview**

This repository contains a 3-part academic project that explores **data collection, analysis, and modeling of Greek Supreme Court (Areios Pagos) decisions**.  
The main goals are:
- To **scrape and structure** judicial decisions into a clean dataset  
- To **classify and analyze** legal texts using **Machine Learning**  
- To **cluster and summarize** decisions with **LLMs and NLP techniques**

---

## Table of Contents

- [Part 1 — Web Crawling & Data Collection](#part-1--web-crawling--data-collection)
- [Part B1 — Legal Document Classification](#part-b1--legal-document-classification)
- [Part B2 — Legal Topic Analysis (Clustering & LLM Titles)](#part-b2--legal-topic-analysis-clustering--llm-titles)
- [Technologies Used](#technologies--tools-used)
- [Results Summary](#results-summary)

---

## Part 1 — Web Crawling & Data Collection

### **Objectives**
Development of a robust web-scraping pipeline to collect judicial decisions from the  
[Areios Pagos official website](https://areiospagos.gr).

#### Goals
1. **Dataset Specification** – Define structure based on [GreekLegalSum](https://huggingface.co/datasets/DominusTea/GreekLegalSum)  
2. **Web Crawling & Scraping** – Extract all Criminal & Civil division decisions  
3. **Data Comparison** – Compare against GreekLegalSum  
4. **Visualization** – Summarize findings with charts and metrics  

---

### **Crawlers**

#### Crawler 1 — General Decision Collection
- Scrapes all decisions across all available years  
- Extracts decision number, type, year, text  
- Saves structured data to CSV  
- Ensures uniqueness (type + decision number)

#### Crawler 2 — Year-Focused (2024)
- Filters decisions for **year 2024** only  
- Performs form-based queries on the Areios Pagos site  
- Collects metadata and decision text  

| Feature | Crawler 1 | Crawler 2 |
|----------|------------|-----------|
| Scope | All years | Only 2024 |
| Target | All categories | Year-based filtering |
| Output | ~20K records | ~1K records |
| Purpose | Comprehensive dataset | Focused analysis |

---

### **Data Analysis**

| Dataset | Description | Shape |
|----------|--------------|-------|
| `df_hugging` | GreekLegalSum (Hugging Face) | 8,395 × 5 |
| `df` | Scraped Areios Pagos data (all years) | 20,795 × 14 |
| `df_final` | Combined & cleaned dataset | — |

- The new dataset contains **richer metadata** and **fewer nulls** than GreekLegalSum.  
- Fully prepared for downstream **AI/NLP tasks** such as summarization or classification.

---

### **Results**
-  Full automation of data extraction pipeline  
-  Clean dataset of 20K+ decisions  
-  Direct comparison and improvement upon GreekLegalSum  

---

## Part B1 — Legal Document Classification

### **Goal**
To build and compare **supervised text classification models** that predict the thematic category of Greek legal documents from the  
[Greek Legal Code dataset](https://huggingface.co/datasets/AI-team-UoA/greek_legal_code).

---

### **Dataset**
- 47K legal documents  
- Hierarchical structure:
  - **Volume**
  - **Chapter**
  - **Subject**
- Data provided in train/validation/test Parquet files  

---

### **Models Implemented**

#### 1. **SVM (LinearSVC)**
- Representations: **BoW** and **TF-IDF**
- Tuned hyperparameters: `C`, `ngram_range`, `min_df`, `max_df`
- **TF-IDF outperformed BoW**

#### 2. **Logistic Regression (Embeddings)**
- Embeddings from:
  - Custom **Word2Vec** (trained on legal data)
  - Pretrained **fastText cc.el.300.bin**
- Used **TF-IDF weighting** and **oversampling**  
- Best overall results on **subject classification**

#### 3. **Multi-Layer Perceptron (MLP)**
- Implemented in both `scikit-learn` and `PyTorch`  
- Trained on both **TF-IDF** and **embeddings**  
- GPU acceleration via **Google Colab Pro**  
- **TF-IDF MLPs > embedding-based MLPs**

---

### **Results Summary**

| Model | Representation | Dataset | Performance |
|--------|----------------|----------|--------------|
| SVM | TF-IDF | Volume | Best classical baseline |
| Logistic Regression | Word2Vec (trained) | Subject | Best overall |
| MLP (TF-IDF) | TF-IDF | Chapter | Best deep model |

> 🔍 Exact metrics (Accuracy, Precision, Recall, F1) available in `B1.ipynb`.

---

### **Insights**
- TF-IDF remains extremely strong for Greek legal text.
- Domain-trained embeddings outperform generic pretrained ones.
- OOV rate (~70%) limits model generalization.
- Oversampling helps minimally due to extreme class imbalance.
- GPU MLPs efficiently scale to large data.

---

## Part B2 — Legal Topic Analysis (Clustering & LLM Titles)

### **Overview**
This part explores **unsupervised learning and topic discovery** from the  
**GreekLegalSum** dataset.  
Conducted in **Google Colab Pro** using a combination of:
`pandas`, `numpy`, `matplotlib`, `seaborn`,  
`scikit-learn`, `optuna`, `sentence-transformers`, `unsloth`, and `openai`.

---

### **Dataset**
- 8,395 Areios Pagos decisions  
- Fields: `text`, `summary`, `case_category`, `case_tags`
- ~25% missing values in `case_category` and `case_tags`

---

### **Part i — Exploratory Data Analysis**
- 399 unique `case_category` values  
- 997 unique `case_tags`  
- Created frequency plots & heatmaps to analyze tag relationships  

**Example findings:**
- *“Ακυρότητα απόλυτη”* often co-occurs with *“Αιτιολογίας επάρκεια”* and *“Νόμου εφαρμογή και ερμηνεία”*  
- Concentration around few dominant themes

---

### **Part ii — Clustering (K-Means)**
1. Started with **TF-IDF**, but optimal `K=2` (too low)  
2. Switched to **Sentence Embeddings** for semantic representation  
3. **Optimized K with Optuna** (based on silhouette score)  
4. **PCA applied** for dimensionality reduction  
5. Final **K = 5** clusters — thematically coherent  

**Evaluation metrics:**
- Silhouette (macro/micro)
- NMI (Normalized Mutual Information)

> Macro silhouette gave the most stable, meaningful clusters.

---

### **Part iii — Title Generation with LLM**
Used an **LLM (Llama-Krikri-8B-Instruct via Unsloth)** to automatically assign **titles** to each cluster.

#### Prompt Template
```python
"Παρακάτω δίνονται τρεις περιλήψεις δικαστικών αποφάσεων. "
"Σκοπός σου είναι να προσδιορίσεις έναν σύντομο, περιεκτικό και νομικά ακριβή τίτλο "
"που να περιγράφει το κοινό θέμα τους.\n\n"
...
"Απόφαση 1:\n{doc1}\n\nΑπόφαση 2:\n{doc2}\n\nΑπόφαση 3:\n{doc3}\n\nΤίτλος:\n"
