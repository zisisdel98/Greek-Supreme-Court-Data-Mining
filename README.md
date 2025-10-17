# Web Crawling & Data Collection of Areios Pagos Decisions (Supreme Court of Greece)

# Project Overview
This project focuses on developing a robust web crawling and scraping pipeline to collect, process, and analyze **legal decisions from the Areios Pagos (Supreme Court of Greece)**.  
The main objective was to build a **structured and comprehensive dataset** of judicial decisions from **both Criminal and Civil Divisions**, focusing primarily on the **year 2024**, but extended to include all available decisions for completeness.

This work was developed as part of an academic assignment, aiming to extract meaningful legal data, compare it against an existing dataset (GreekLegalSum), and visualize the results.

---

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

