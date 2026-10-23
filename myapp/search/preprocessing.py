"""
Text pre-processing for the fashion products dataset (IRWA Part 1).
The same build_terms function is used for the documents and for the queries.
"""
import re
import unicodedata
from urllib.parse import urlparse

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download the stop word list the first time
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# fields we have to return for every document
OUTPUT_FIELDS = ["pid", "title", "description", "brand", "category", "sub_category", "product_details",
                 "seller", "out_of_stock", "selling_price", "discount", "actual_price", "average_rating", "url"]

# text fields, each one gets its own list of terms ("title_terms", "brand_terms", ...)
TEXT_FIELDS = ["title", "description", "brand", "category", "sub_category", "product_details", "seller"]

# numeric / true-false fields: we keep them as numbers and do not make terms out of them
NUMERIC_FIELDS = ["out_of_stock", "selling_price", "discount", "actual_price", "average_rating"]

# product_details keys that are just product codes, not useful text
DETAILS_EXCLUDED_KEYS = {"Style Code"}


def repair(match):
    """Turn one piece of broken text back into the right character (used by fix_mojibake)."""
    try:
        return match.group().encode("latin-1").decode("utf-8")
    except UnicodeError:
        return match.group()  ## could not repair it, leave it as it is


def fix_mojibake(text):
    """Fix characters that were broken when the data was saved ("youâ\\x80\\x99re" -> "you’re")."""
    # some texts were broken twice, so we try up to 3 times
    for i in range(3):
        new_text = re.sub(r"[\xc2-\xf4][\x80-\xbf]+", repair, text)
        if new_text == text:
            break
        text = new_text
    return text


def normalize_text(text):
    """Fix broken characters, remove accents, replace apostrophes and other special characters by spaces, lowercase."""
    if text is None:
        text = ""
    text = fix_mojibake(text)
    text = unicodedata.normalize("NFKD", text)  ## "é" becomes "e" + accent
    text = "".join([ch for ch in text if not unicodedata.combining(ch)])  ## remove the accents
    text = re.sub(r"['‘’‛ʼ`´]", " ", text)  ## apostrophes become spaces ("men's" -> "men s")
    text = "".join([ch if ch.isascii() else " " for ch in text])  ## other special characters become spaces
    text = text.lower()
    return text


def tokenize(text):
    """Normalize the text and split it into words (letters and digits only)."""
    text = normalize_text(text)
    text = re.sub(r"\b([a-z])-(?=[a-z]{2,}\b)", r"\1", text)  ## "t-shirt" -> "tshirt", "v-neck" -> "vneck"
    text = re.sub(r"\b([tv])\s+(?=(shirt|shirts|neck)\b)", r"\1", text)  ## "t shirt" -> "tshirt", "v neck" -> "vneck"
    tokens = re.findall(r"[a-z0-9]+", text)  ## every other character (punctuation, spaces) splits words
    return tokens


def build_terms(text, remove_stopwords=True, stem=True):
    """
    Preprocess a text (a document field or a query) and return its list of terms.

    Argument:
    text -- string to preprocess
    remove_stopwords, stem -- only set to False to show the intermediate steps

    Returns:
    terms -- list of terms, in the same order as in the text
    """
    terms = tokenize(text)  ## lowercase, remove punctuation, split into words
    terms = [term for term in terms if not term.isdigit()]  ## remove numbers
    terms = [term for term in terms if len(term) > 1]  ## remove 1-letter words
    if remove_stopwords:
        terms = [term for term in terms if term not in stop_words]  ## remove stop words
    if stem:
        terms = [stemmer.stem(term) for term in terms]  ## stemming
    return terms


def parse_product_details(details):
    """Turn the list of small dicts [{"Fabric": "Cotton"}, {"Fit": "Slim"}, ...] into one dict."""
    result = {}
    no_name_values = []
    for item in details:
        for key, value in item.items():
            key = key.strip()
            value = str(value).strip()
            if value == "":
                continue
            if key == "":
                no_name_values.append(value)  ## some entries have no attribute name
            elif key not in result:
                result[key] = value
    if len(no_name_values) > 0:
        result["Other"] = "; ".join(no_name_values)
    return result


def details_to_text(details):
    """Text used for the product_details terms: only the values, without product codes and "No" values."""
    words = []
    for key, value in details.items():
        if key in DETAILS_EXCLUDED_KEYS:
            continue
        if value.lower() == "yes":
            words.append(key)  ## "Hooded": "Yes" -> "Hooded"
        elif value.lower() in ["no", "na", "n/a", "none"]:
            continue
        else:
            words.append(value)
    return " ".join(words)


def recover_brand(brand, url):
    """
    The brand field is often cut off ("REEB", "Pu"), but the url starts with the full brand
    ("reebok-full-sleeve-..."). Returns the full brand in lowercase, or None if it does not match.
    """
    brand_words = re.findall(r"[a-z0-9]+", normalize_text(brand))
    brand_words = [word for word in brand_words if word not in ["and", "by", "the", "of"]]  ## urls skip these words
    if len(brand_words) == 0 or url is None:
        return None
    slug = urlparse(url).path.strip("/").split("/")[0]  ## e.g. "reebok-full-sleeve-solid-men-sweatshirt"
    slug_words = re.findall(r"[a-z0-9]+", normalize_text(slug))
    n = len(brand_words)
    if len(slug_words) < n:
        return None
    if slug_words[:n - 1] != brand_words[:n - 1]:  ## every brand word except the last one must be the same
        return None
    if not slug_words[n - 1].startswith(brand_words[n - 1]):  ## the last one can be cut off ("reeb" -> "reebok")
        return None
    return " ".join(slug_words[:n])


def parse_number(value):
    """ "2,999" -> 2999.0, "69% off" -> 69.0, "3.9" -> 3.9, "" -> None"""
    if value is None:
        return None
    value = value.replace(",", "")
    match = re.search(r"\d+(\.\d+)?", value)
    if match is None:
        return None
    return float(match.group())


def clean_text(value):
    """Remove spaces at the start and end; an empty text becomes None (missing)."""
    value = value.strip()
    if value == "":
        return None
    return value


def process_document(raw):
    """Preprocess one document of the dataset and return it as a dict."""
    doc = {}
    # 1. the fields we have to return
    doc["pid"] = raw["pid"]
    doc["title"] = clean_text(raw["title"])
    doc["description"] = clean_text(raw["description"])
    doc["brand"] = clean_text(raw["brand"])
    doc["category"] = clean_text(raw["category"])
    doc["sub_category"] = clean_text(raw["sub_category"])
    doc["product_details"] = parse_product_details(raw["product_details"])
    doc["seller"] = clean_text(raw["seller"])
    doc["out_of_stock"] = bool(raw["out_of_stock"])
    doc["selling_price"] = parse_number(raw["selling_price"])
    doc["discount"] = parse_number(raw["discount"])
    doc["actual_price"] = parse_number(raw["actual_price"])
    doc["average_rating"] = parse_number(raw["average_rating"])
    doc["url"] = clean_text(raw["url"])

    # 2. full brand name (from the url, otherwise the brand field)
    brand_full = recover_brand(doc["brand"], doc["url"])
    if brand_full is None and doc["brand"] is not None:
        brand_full = " ".join(re.findall(r"[a-z0-9]+", normalize_text(doc["brand"])))
    if brand_full == "":
        brand_full = None
    doc["brand_full"] = brand_full

    # 3. one list of terms per text field, plus all of them together
    texts = {}
    for field in TEXT_FIELDS:
        texts[field] = doc[field]
    texts["brand"] = doc["brand_full"]
    texts["product_details"] = details_to_text(doc["product_details"])

    all_terms = []
    for field in TEXT_FIELDS:
        terms = build_terms(texts[field])
        doc[field + "_terms"] = terms
        all_terms.extend(terms)
    doc["all_terms"] = all_terms
    return doc


def preprocess_corpus(raw_docs):
    """
    Preprocess all the documents.

    Argument:
    raw_docs -- list of documents as loaded from fashion_products_dataset.json

    Returns:
    df -- DataFrame with one row per document, indexed by pid
    """
    rows = []
    for raw in raw_docs:
        rows.append(process_document(raw))
    df = pd.DataFrame(rows)
    df = df.set_index("pid", drop=False)  ## so we can do df.loc[pid]
    return df


def load_processed_corpus(path):
    """Load the processed corpus saved by the Part 1 notebook (data/processed_corpus.jsonl)."""
    df = pd.read_json(path, orient="records", lines=True, dtype={"pid": str})
    df = df.set_index("pid", drop=False)
    return df
