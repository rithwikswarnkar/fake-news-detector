# 📰 Fake News Detection Project

## Overview

This project explores **fake news detection using Natural Language Processing (NLP)** techniques. Using a public dataset of labeled fake and true news articles, we build classification models to distinguish between them.

The project emphasizes:

* **Data cleaning** (removing junk, duplicates, and tweet-like content)
* **Baseline modeling** with logistic regression + TF-IDF
* **Data leakage testing** (identifying the `subject` column as a leak)
* **Stress-testing** model generalization with alternative train/test splits
* **Feature importance analysis** to investigate what the model is really learning

---

## Dataset

* Source: [Fake News Dataset on Kaggle](https://www.kaggle.com/datasets/abaghyangor/fake-news-dataset)
* Contains two files:

  * **Fake.csv** – articles labeled as fake
  * **True.csv** – articles labeled as true

Columns:

* `title` → Article title
* `text` → Article body text
* `subject` → Topic/source (later removed due to leakage)
* `date` → Publication date

---

## Data Cleaning Steps

To improve data quality and avoid misleading results, the following steps were taken:

1. **Removed leakage column**

   * Dropped `subject` after confirming it perfectly revealed the label.

2. **Removed invalid content**

   * Empty articles
   * Articles containing only URLs
   * Very short texts (< 20 words)
   * Titles with fewer than 3 words

3. **Removed duplicates**

   * Deduplicated by both `title` and `text`.

4. **Filtered noisy “tweet-like” content**

   * Articles resembling social media posts (hashtags, @mentions, slang, excessive caps) were dropped.

---

## Modeling Approach

### Feature Engineering

* **TF-IDF vectorization** applied separately on:

  * **Titles** (up to 5,000 features, 1–2 grams)
  * **Texts** (up to 10,000 features, unigrams only)
* Features combined into a sparse representation.

### Classifier

* **Logistic Regression** (`max_iter=2000`)
* Evaluated with classification reports + confusion matrices

---

## Experiments

### Baseline Models

* **Model A (subject only):** 100% accuracy → data leakage.
* **Model B (title only):** ~94% accuracy.
* **Model C (text only):** ~98% accuracy.

### Final Pipeline

* Combined **title + text TF-IDF features**
* Random 80/20 split for baseline testing
* “Source-based split” (alphabetical by title) as a stress-test for unseen publishers

---

## Results

* **Random split:** Achieved ~98.8% accuracy.
* **Source-based split:** Performance dropped but still high.
* **Feature analysis:**

  * Random split learned **sensationalized words** and rare patterns.
  * Source split learned **publisher-specific tokens** (`reuters`, `factbox`, weekdays, etc.).
  * **0% overlap** in top features → strong evidence of source-specific memorization.

---

## Key Insights & Limitations

⚠️ **Critical finding:** Models are not learning “truthfulness,” but rather **publisher/style artifacts**.

* Extremely high accuracy (98%+) is misleading.
* Coefficients highlight publication identifiers rather than semantic content.
* Results do not generalize well outside this dataset.

---

## Next Steps

* Train/test splits by **source** rather than random sampling.
* Evaluate on **external datasets** to test generalization.
* Explore **domain adaptation** or adversarial training to reduce reliance on publisher-specific signals.
* Consider more advanced models (e.g., transformers) but **focus on dataset quality first**.

---

## Tech Stack

* **Python** 3.10
* **Pandas** for data processing
* **Matplotlib/Seaborn** for visualization
* **Scikit-learn** for ML pipeline (TF-IDF, Logistic Regression, metrics)
* **SciPy** for sparse matrix handling

---

## Project Structure

```
fake-news-detector/
│── data/
│   ├── Fake.csv
│   ├── True.csv
│── notebooks/
│   ├── exploration.ipynb
│   ├── cleaning.ipynb
│   ├── modeling.ipynb
│── README.md
```

---

## Lessons Learned

* **Dataset bias** can inflate performance.
* **Data cleaning and leakage detection** are just as important as model design.
* Even simple models (logistic regression + TF-IDF) can perform extremely well on biased datasets—but this may not reflect real-world difficulty.