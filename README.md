# Web Crawling & Data Collection of Areios Pagos Decisions (Supreme Court of Greece)

# Project Overview
This project focuses on developing a robust web crawling and scraping pipeline to collect, process, and analyze **legal decisions from the Areios Pagos (Supreme Court of Greece)**.  
The main objective was to build a **structured and comprehensive dataset** of judicial decisions from **both Criminal and Civil Divisions**, focusing primarily on the **year 2024**, but extended to include all available decisions for completeness.

This work was developed as part of an academic assignment, aiming to extract meaningful legal data, compare it against an existing dataset (GreekLegalSum), and visualize the results.

---

# Part 1

# Objectives
**Data Collection of Areios Pagos Decisions**

1. **Data Specification Extraction**  
   Define the schema of the dataset based on the [GreekLegalSum dataset on Hugging Face](https://huggingface.co/datasets/DominusTea/GreekLegalSum).

2. **Web Crawling & Scraping Implementation**  
   Develop and execute multiple crawlers to extract decisions from [https://areiospagos.gr](https://areiospagos.gr).

3. **Data Comparison & Analysis**  
   Compare the collected data with GreekLegalSum to evaluate coverage, quality, and completeness.

4. **Data Visualization**  
   Present findings through clear and justified visualizations.

---

## Technical Approach

### Crawler 1: Decision Collection by Case Type
- Navigates through the main decision search page.  
- Extracts all available decision types from dropdown menus.  
- Visits each decision page and scrapes HTML content (`<p>` tags).  
- Saves the data (decision number, year, type, full text) into a CSV file.  
- Prevents duplicates via a unique key (case type + decision number).  

### Crawler 2: Targeted Collection for 2024
- Focused exclusively on decisions issued in **2024**.  
- Performs form-based searches using the year field.  
- Extracts decision links from search results and scrapes their content.  
- Stores results locally as a CSV file.  

### Comparison of Crawlers
| Feature | Crawler 1 | Crawler 2 |
|----------|------------|-----------|
| Scope | All years | Only 2024 |
| Target | All decision categories | Year-based filtering |
| Output | ~20K records | ~1K records |
| Goal | Comprehensive dataset | Focused dataset for analysis |

---

## Data Analysis

### Datasets Used
1. **`df_hugging`** – Original dataset from GreekLegalSum (Hugging Face).  
   - Shape: (8,395 × 5)  
   - Columns: `text`, `summary`, `case_category`, `case_tags`, `subset`  
   - Contains missing values (especially in `case_category` and `case_tags`).  

2. **`df`** – Data collected by the first crawler (all years).  
   - Shape: (20,795 × 14)  
   - Very few null values.  
   - Richer metadata and detailed structure.  

3. **`df_final`** – Combined dataset (`df` + `df_2024`).  
   - Complete, cleaned, and analysis-ready dataset.  

### Key Findings
- The **collected data** is more detailed and complete than GreekLegalSum.  
- Each decision includes metadata such as division type, year, number, and referenced legal articles.  
- The **final dataset** is suitable for downstream **AI/NLP applications** such as text classification or summarization.  

---

## Technologies & Tools Used
| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python 3 |
| **Web Crawling & Scraping** | `requests`, `BeautifulSoup`, `Selenium`, `re`, `time` |
| **Data Handling** | `pandas`, `numpy`, `csv` |
| **Visualization** | `matplotlib`, `plotly`, `seaborn` (optional) |
| **Data Source** | [Areios Pagos official site](https://areiospagos.gr) |
| **Reference Dataset** | [GreekLegalSum (Hugging Face)](https://huggingface.co/datasets/DominusTea/GreekLegalSum) |

---

## Results Summary
- Built a **fully automated data collection pipeline** for Greek Supreme Court decisions.  
- Generated a **clean, extensive dataset (20K+ records)** suitable for data analysis and ML applications.  
- Demonstrated **data engineering, scraping, and analytical visualization** skills.  
- Compared and improved upon an existing open dataset (GreekLegalSum).  

---

# Β1

# Legal Document Classification using Supervised Machine Learning

This project implements and compares multiple **text classification models** for predicting the category of **Greek legal documents**.  
It was developed as part of the *Data Mining* course at the University of Athens and demonstrates practical expertise in **Natural Language Processing (NLP)**, **feature engineering**, and **model evaluation**.

---

## Overview

The goal is to automatically classify legal texts from the [Greek Legal Code dataset](https://huggingface.co/datasets/AI-team-UoA/greek_legal_code), which contains **47k legal documents** categorized by:
- **Collection (Volume)**
- **Chapter**
- **Subject**

Each document is associated with one label from each of the above hierarchies.  
The challenge includes **high class imbalance**, **few-shot/zero-shot categories**, and a **large OOV ratio** in the test sets.

---

## Dataset

The dataset comes from the *Raphtarchis* collection and is split into:
- `train`, `validation`, and `test` subsets  
- Three classification levels:
  - `volume`
  - `chapter`
  - `subject`

Examples are stored in Parquet format and loaded directly from Hugging Face.

---

## Models & Approach

Three main classification approaches were implemented and compared:

### ** 1. SVM with BoW and TF-IDF**
- Model: `LinearSVC`
- Representations: Bag-of-Words and TF-IDF
- Tuned hyperparameters: `C`, `ngram_range`, `min_df`, `max_df`
- Evaluation across all three datasets (volume, chapter, subject)
- **TF-IDF consistently outperformed BoW**, due to term-frequency weighting

### ** 2.  Logistic Regression with Dense Embeddings**
- Word embeddings generated from:
  - `Word2Vec` (trained from scratch on legal data)
  - `fastText` (pretrained `cc.el.300.bin`, with and without fine-tuning)
- Tokenization via `spaCy` (`el_core_news_sm`)
- Optional **TF-IDF weighting** applied to embedding vectors
- **Oversampling** applied to underrepresented classes
- Fine-tuned fastText and trained Word2Vec models produced the best results on `subject` data

### ** 3. Multi-Layer Perceptron (MLP)**
- Implemented both with:
  - `scikit-learn` (TF-IDF representation)
  - `PyTorch` (GPU-accelerated for large, multi-label datasets)
- Compared performance using both TF-IDF and word embeddings
- **TF-IDF-based MLPs outperformed embedding-based ones**
- Used GPU runtime (Google Colab Pro) for efficient training on large data

---

## Implementation Details

| Component | Library / Tool |
|------------|----------------|
| Data Loading | `pandas`, `pyarrow` |
| Text Preprocessing | `re`, `spaCy`, Greek stopwords filtering |
| Feature Extraction | `TfidfVectorizer`, custom Word2Vec / fastText embeddings |
| Models | `scikit-learn` (SVM, Logistic Regression, MLP), `PyTorch` (MLP) |
| Sampling | Custom oversampling for few-shot classes |
| Evaluation | Accuracy, Precision, Recall, F1-score |
| Environment | Google Colab Pro (GPU enabled) |

---

## Results Summary

| Model | Representation | Dataset | Accuracy | Precision | Recall | F1-score |
|--------|----------------|----------|-----------|------------|----------|-----------|
| SVM | TF-IDF | Volume | ↑ Best among baselines |  |  |  |
| Logistic Regression | Word2Vec (trained) | Subject | ↑ Best overall |  |  |  |
| MLP (TF-IDF) | TF-IDF | Chapter | ↑ Best deep model |  |  |  |

> *Exact scores can be found in `B1.ipynb` under the results section.*

---

## Κey Insights

- **TF-IDF** remains highly effective for legal text classification.
- **Word2Vec trained on domain data** outperformed pretrained embeddings for specialized vocabulary.
- **High OOV ratios** (≈70%) significantly impacted model generalization for `chapter` and `subject` levels.
- **Oversampling** offered minor improvements due to severe class imbalance.
- **GPU-accelerated PyTorch MLP** allowed efficient handling of large, multi-label data.

---




