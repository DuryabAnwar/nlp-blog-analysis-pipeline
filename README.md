# Blog Content Mining & NLP Analysis Pipeline

> **NED University of Engineering & Technology**
> Course: CT-377 — Data Mining | Fall 2025

An end-to-end NLP pipeline that scrapes blog content from multiple sources, analyzes tone, motive, and sentiment, generates AI comments using GPT-2, and visualizes insights — all driven by a single user-defined keyword.

---

## Project Overview

Given any keyword (e.g., *"Machine Learning"*, *"Cybersecurity"*), this pipeline:

1. **Scrapes** articles from Wikipedia, Dev.to, and HackerNews
2. **Extracts keywords** using TF-IDF
3. **Models topics** using Latent Dirichlet Allocation (LDA)
4. **Detects motive & tone** of each article (Tutorial, Academic, Promotional, etc.)
5. **Generates user comments** using GPT-2 via HuggingFace Transformers
6. **Analyzes sentiment** on both scraped and AI-generated comments
7. **Visualizes** everything with 5 charts + a word cloud

---

## Tech Stack

| Category | Libraries |
|---|---|
| Web Scraping | `requests`, `beautifulsoup4` |
| Data Handling | `pandas` |
| NLP | `nltk`, `textblob`, `scikit-learn` (TF-IDF, LDA) |
| LLM / Generative AI | `transformers` (GPT-2 via HuggingFace) |
| Visualization | `matplotlib`, `wordcloud` |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/blog-content-mining.git
cd blog-content-mining
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook DM_CCP_FIXED.ipynb
```

Enter any keyword when prompted (e.g., `Deep Learning`, `Blockchain`, `Climate Change`).

> **Note:** First run downloads the GPT-2 model (~500 MB). Subsequent runs use the cached version.

---

## Limitations & Future Work

- Popularity proxy is simulated — real implementation would pull actual API engagement metrics
- GPT-2 is lightweight; a production system would use a larger instruction-tuned model
- Motive detection uses keyword matching; a fine-tuned classifier would be more robust

---

## Author

**Duryab Anwar** — BS Computer Science, NED University

---

## License
MIT License — see LICENSE for details.
