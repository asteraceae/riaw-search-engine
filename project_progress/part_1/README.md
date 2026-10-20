# Part 1 - Text Processing and Exploratory Data Analysis

Deadline: **23/10/2026, 23:59**. Tag: `IRWA-2026-part-1` (see root README).

## Contents
- `part_1.ipynb` - code for text pre-processing and EDA (skeleton, TODO)
- `IRWA-2026-uXXXXXX-uXXXXXX-uXXXXXX-part-1.pdf` - report (TODO)

## How to reproduce
1. Set up the environment as described in the root `README.md` (`pip install -r requirements.txt`).
2. Put `fashion_products_dataset.json` (and `validation_labels.csv`) in `data/` at the repo root.
3. Download the NLTK resources used (stopwords, and whatever else the notebook needs).
4. Run `part_1.ipynb` top to bottom.

## Requirements checklist (from the brief)
### Data preparation
- [ ] Pre-process `title` and `description`: stop-word removal, tokenization, punctuation removal, stemming (+ anything extra = bonus)
- [ ] Output keeps, when present: pid, title, description, brand, category, sub_category, product_details, seller, out_of_stock, selling_price, discount, actual_price, average_rating, url
- [ ] Decide + justify handling of category, sub_category, brand, product_details, seller (merge into one text field vs separate fields vs other; pros/cons, distinctiveness)
- [ ] Decide + justify handling of out_of_stock, selling_price, discount, actual_price, average_rating (index as text terms or not)
- [ ] Keep `pid` (used for evaluation)
- [ ] Look at `validation_labels.csv` (query_1: "women full sleeve sweatshirt cotton", query_2: "men slim jeans blue") to inform field processing/weighting

### Exploratory data analysis
- [ ] Word count distribution, average sentence length, vocabulary size
- [ ] Rankings: rating, price, discount, top sellers and brands
- [ ] out_of_stock distribution
- [ ] Word clouds of most frequent words
- [ ] Entity recognition
- [ ] Explain the decisions made for the analysis

### Report
- [ ] Decisions, assumptions, GitHub URL + tag
- [ ] AI-usage acknowledgement (required by the course policy)
