# Part 1 - Text Processing and Exploratory Data Analysis

Deadline: **23/10/2026, 23:59**. Tag: `IRWA-2026-part-1` (see root README).

## Contents
| Path | Description |
|---|---|
| `../../myapp/search/preprocessing.py` | Pre-processing functions (`build_terms`, `preprocess_corpus`, `load_processed_corpus`). Later parts import them so that documents and queries go through the same pipeline. |
| `part_1.ipynb` | Applies the pre-processing to the corpus, saves the processed corpus, inspects `validation_labels.csv` and runs the EDA. Executed, outputs included. |
| `figures/*.png` | Every plot of the notebook (regenerated on each run). |
| `output/processed_documents_sample.json` | First 20 processed documents, to show the output schema without running anything. |
| `output/ner_entities_sample.csv` | Entities found by spaCy in the NER sample (pid, text, label). |
| `../../data/processed_corpus.jsonl` | Full processed corpus, one document per line (written by the notebook; `data/` is git-ignored). |
| `IRWA-2026-uXXXXXX-uXXXXXX-uXXXXXX-part-1.pdf` | Report (TODO, written separately). |

## Setup
Tested with Python 3.12. From the repository root:
```bash
uv venv .venv --python 3.12            # or: python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt     # or: pip install -r requirements.txt
```
`requirements.txt` contains the template packages plus what Part 1 needs: `matplotlib`, `wordcloud`, `spacy` and its
English model `en_core_web_sm` (installed from its GitHub release URL), and `jupyter` / `nbconvert` to run the notebook.
The NLTK stop-word list is downloaded automatically the first time `myapp.search.preprocessing` is imported
(internet needed once). To download it by hand: `python -c "import nltk; nltk.download('stopwords')"`.

## How to reproduce
1. Put `fashion_products_dataset.json` and `validation_labels.csv` in `data/` at the repository root.
2. Execute the notebook top to bottom, from the repository root:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace project_progress/part_1/part_1.ipynb
   ```
   or open it with `jupyter notebook project_progress/part_1/part_1.ipynb` and run all cells.
   Paths are relative to the notebook's own folder, which is the kernel's working directory in Jupyter, nbconvert
   and VS Code. A full run takes about one minute.
3. Results: figures in `figures/`, the sample files in `output/`, and the full processed corpus in
   `data/processed_corpus.jsonl`. All samples use `RANDOM_STATE = 42`, so reruns give the same numbers.

## Using the pre-processing in later parts
```python
from myapp.search.preprocessing import build_terms, load_processed_corpus

docs = load_processed_corpus("data/processed_corpus.jsonl")  # DataFrame indexed by pid
query_terms = build_terms("men slim jeans blue")              # ['men', 'slim', 'jean', 'blue']
```
`build_terms(text, remove_stopwords=True, stem=True)`: the two flags only exist to show intermediate steps;
documents and queries use the defaults.

### Pipeline (`build_terms`)
1. repair mojibake (UTF-8 text decoded as Latin-1, e.g. `youâ\x80\x99re`);
2. fold accents to ASCII; apostrophes and any other non-ASCII character (non-breaking spaces, dashes, symbols) become spaces;
3. lower-case;
4. join single-letter compounds: `t-shirt` / `t shirt` → `tshirt`, `v-neck` / `v neck` → `vneck`, `x-large` → `xlarge`;
5. tokenize on every non-alphanumeric character (removes punctuation, splits `look.Designed`);
6. drop purely numeric tokens and tokens shorter than 2 characters;
7. remove NLTK English stop words;
8. Porter stemming.

### Processed document schema
- The 14 output fields, in this order: `pid` (unchanged), `title`, `description`, `brand`, `category`, `sub_category`,
  `product_details` (dict; entries with an empty attribute name are kept under `"Other"`), `seller`,
  `out_of_stock` (bool), `selling_price`, `discount` (percentage), `actual_price`, `average_rating` (floats), `url`.
  Missing values are `null`; empty strings are mapped to `null`. Every record validates against
  `myapp.search.objects.Document`.
- `brand_full`: lower-case brand recovered from the URL (falls back to the normalized `brand` field).
- One term list per text field: `title_terms`, `description_terms`, `brand_terms` (from `brand_full`),
  `category_terms`, `sub_category_terms`, `product_details_terms`, `seller_terms`.
  `product_details_terms` index the attribute values, not the attribute names; `Style Code` is skipped; Yes/No
  attributes index the attribute name only when the value is Yes (`"Hooded": "Yes"` → `hooded`).
- `all_terms`: concatenation of the seven term lists.
- `out_of_stock`, `selling_price`, `discount`, `actual_price`, `average_rating` are not turned into terms.

## Notebook sections
0. Setup · 1. Load the corpus (missing values) · 2. Text pre-processing (examples, step-by-step
view) · 3. Other text fields (`product_details`, brand recovery, whole
corpus, output-field checks, saving) · 4. Numeric and boolean fields · 5. Validation labels (query terms per field
and label) · 6. EDA: word counts, sentence length, vocabulary (per field, top terms, bigrams), word clouds,
categories, brands and sellers, prices/discount/rating, product rankings, out of stock, named-entity recognition
(spaCy) and structured attributes · 7. Saved files.

## Requirements checklist (from the brief)
### Data preparation
- [x] Pre-process `title` and `description`: stop-word removal, tokenization, punctuation removal, stemming (+ extras: mojibake repair, accent folding, compound joining, numeric / short token removal)
- [x] Output keeps, when present: pid, title, description, brand, category, sub_category, product_details, seller, out_of_stock, selling_price, discount, actual_price, average_rating, url
- [x] Handling of category, sub_category, brand, product_details, seller: separate term list per field + configurable merged `all_terms` (justification: report)
- [x] Handling of out_of_stock, selling_price, discount, actual_price, average_rating: parsed to bool / float, not indexed as terms (justification: report)
- [x] Keep `pid` (used for evaluation)
- [x] Look at `validation_labels.csv` (query_1: "women full sleeve sweatshirt cotton", query_2: "men slim jeans blue"): notebook section 5

### Exploratory data analysis
- [x] Word count distribution, average sentence length, vocabulary size
- [x] Rankings: rating, price, discount, top sellers and brands
- [x] out_of_stock distribution
- [x] Word clouds of most frequent words
- [x] Entity recognition
- [ ] Explain the decisions made for the analysis (report)

### Report
- [ ] Decisions, assumptions, GitHub URL + tag
- [ ] AI-usage acknowledgement (required by the course policy)
