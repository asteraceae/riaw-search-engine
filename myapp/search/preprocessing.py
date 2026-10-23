"""
Text pre-processing for the fashion products corpus (IRWA Part 1).

Documents and queries must go through exactly the same pipeline, so every later part (indexing, ranking, the web
app) should import `build_terms` / `preprocess_query` from this module instead of re-implementing them.

Pipeline of `build_terms` (title, description and the other text fields):
    1. repair mojibake (UTF-8 text that was decoded as Latin-1, e.g. "youâ\\x80\\x99re" -> "you're")
    2. Unicode folding: accents removed ("café" -> "cafe"), any other non-ASCII character becomes a space
    3. lower-casing
    4. joining of single-letter compounds ("t-shirt", "t shirt" -> "tshirt"; "v-neck", "v neck" -> "vneck")
    5. tokenization on every non-alphanumeric character (removes punctuation and also splits words glued by
       punctuation, e.g. "look.designed")
    6. removal of purely numeric tokens and of tokens shorter than `min_length`
    7. removal of NLTK English stop words
    8. stemming (Porter by default)

Field handling of `preprocess_document`:
    - every text field (title, description, brand, category, sub_category, product_details, seller) gets its own list
      of terms (`<field>_terms`), so later parts can weight fields separately; `all_terms` concatenates a
      configurable subset of them for a single-field index
    - product_details is flattened to a dict; "Style Code" (an identifier) is not indexed, Yes/No attributes index the
      attribute name when the value is "Yes" ("Hooded": "Yes" -> "hooded") and nothing when it is "No"
    - brand_terms are built from the full brand name recovered from the URL slug (the brand field is often truncated)
    - out_of_stock, selling_price, discount, actual_price and average_rating are parsed into bool / float values and
      are NOT turned into terms (they are meant for filtering and ranking)
"""
import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional
from urllib.parse import urlparse

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer

# Fields returned for every selected document (brief, data preparation, point 2)
OUTPUT_FIELDS = ["pid", "title", "description", "brand", "category", "sub_category", "product_details",
                 "seller", "out_of_stock", "selling_price", "discount", "actual_price", "average_rating", "url"]

# Text fields that get their own list of terms ("<field>_terms")
TEXT_FIELDS = ["title", "description", "brand", "category", "sub_category", "product_details", "seller"]

# Numeric / boolean fields: parsed, kept as values, never turned into terms
NUMERIC_FIELDS = ["out_of_stock", "selling_price", "discount", "actual_price", "average_rating"]

# product_details keys whose values are identifiers rather than descriptive text
DETAILS_EXCLUDED_KEYS = {"Style Code"}

# Key used for product_details entries that have an empty attribute name in the raw data
DETAILS_UNNAMED_KEY = "Other"

# product_details values that only say whether the attribute applies (e.g. "Hooded": "Yes")
DETAILS_YES_VALUES = {"yes"}
DETAILS_NO_VALUES = {"no", "na", "n/a", "none"}

# Words that appear in brand names but are dropped from the URL slugs ("JACK AND HAR" -> "jack-hardy-...")
BRAND_CONNECTOR_WORDS = {"and", "by", "the", "of"}

# Runs of characters that look like UTF-8 bytes decoded as Latin-1 (a lead byte followed by continuation bytes)
_MOJIBAKE_RE = re.compile(r"[\xc2-\xf4][\x80-\xbf]+")
# Apostrophes and quote-like characters: replaced by a space ("men's" -> "men s", the "s" is dropped later)
_APOSTROPHE_RE = re.compile(r"['‘’‛ʼ`´]")
# Single-letter compounds: "t-shirt" -> "tshirt", "x-large" -> "xlarge" (hyphen), and the frequent spaced forms
# "t shirt" / "v neck" (the corpus writes them both ways: "T-Shirt" and "T Shirt", "V-Neck" and "V Neck")
_HYPHEN_COMPOUND_RE = re.compile(r"\b([a-z])-(?=[a-z]{2,}\b)")
_SPACED_COMPOUND_RE = re.compile(r"\b([tv])\s+(?=(?:shirts?|neck)\b)")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _ensure_nltk_resources():
    """Download the NLTK stop word list if it is not available yet (needs internet the first time)."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


_ensure_nltk_resources()
STOP_WORDS = frozenset(stopwords.words("english"))
_STEMMERS = {"porter": PorterStemmer(), "snowball": SnowballStemmer("english")}


@lru_cache(maxsize=None)
def _stem(word: str, stemmer: str) -> str:
    """Stem one word; cached because the vocabulary is much smaller than the number of tokens."""
    return _STEMMERS[stemmer].stem(word)


def fix_mojibake(text: str) -> str:
    """
    Repair UTF-8 text that was wrongly decoded as Latin-1, which happens in some descriptions
    ("youâ\\x80\\x99re" -> "you’re"). Each suspicious run is re-encoded as Latin-1 and decoded as UTF-8; runs that do
    not decode are left untouched. Repeated up to three times because some strings were double-encoded.
    :param text: raw text
    :return: repaired text
    """
    def repair(match):
        try:
            return match.group().encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return match.group()

    for _ in range(3):
        fixed = _MOJIBAKE_RE.sub(repair, text)
        if fixed == text:
            break
        text = fixed
    return text


def normalize_text(text: Optional[str]) -> str:
    """
    Repair mojibake, fold accents to ASCII ("café" -> "cafe"), turn apostrophes and every other non-ASCII character
    (non-breaking spaces, dashes, symbols) into spaces, and lower-case.
    Replacing instead of deleting avoids gluing words such as "reason—several".
    :param text: raw text (None is treated as empty)
    :return: normalized text (ASCII, lower case)
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", fix_mojibake(str(text)))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _APOSTROPHE_RE.sub(" ", text)
    text = "".join(ch if ch.isascii() else " " for ch in text)
    return text.lower()


def tokenize(text: str) -> List[str]:
    """
    Split normalized text into alphanumeric tokens. Every non-alphanumeric character (punctuation, symbols,
    whitespace) is a separator, so punctuation is removed and words glued by punctuation ("look.designed") are split.
    Single-letter compounds are joined first ("t-shirt"/"t shirt" -> "tshirt", "v-neck"/"v neck" -> "vneck"),
    otherwise the letter would be dropped and "t-shirt" would become "shirt".
    :param text: normalized text (see normalize_text)
    :return: list of tokens, in order of appearance
    """
    text = _HYPHEN_COMPOUND_RE.sub(r"\1", text)
    text = _SPACED_COMPOUND_RE.sub(r"\1", text)
    return _TOKEN_RE.findall(text)


def build_terms(text: Optional[str], remove_stopwords: bool = True, stemmer: Optional[str] = "porter",
                keep_numbers: bool = False, min_length: int = 2) -> List[str]:
    """
    Full pre-processing pipeline for a piece of text (document field or query): normalization, tokenization with
    punctuation removal, removal of numeric and very short tokens, stop word removal and stemming.
    Duplicates and order are kept (needed for term frequencies and positional indexes).
    :param text: raw text
    :param remove_stopwords: remove NLTK English stop words
    :param stemmer: "porter", "snowball" or None to skip stemming
    :param keep_numbers: keep purely numeric tokens such as "100" or "2"
    :param min_length: minimum token length (applied before stemming)
    :return: list of terms
    """
    tokens = tokenize(normalize_text(text))
    if not keep_numbers:
        tokens = [t for t in tokens if not t.isdigit()]
    tokens = [t for t in tokens if len(t) >= min_length]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if stemmer:
        tokens = [_stem(t, stemmer) for t in tokens]
    return tokens


def parse_product_details(details) -> Dict[str, str]:
    """
    Turn the raw product_details (a list of single-key dicts) into one dict. Keys and values are stripped and empty
    values dropped. Entries with an empty attribute name are kept under DETAILS_UNNAMED_KEY (joined with "; ").
    If a named key repeats, the first value is kept (in this corpus repeated keys always have the same value).
    :param details: raw product_details value
    :return: dict {attribute: value}
    """
    merged = {}
    unnamed = []
    if isinstance(details, dict):
        details = [details]
    for item in details or []:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            key, value = str(key).strip(), str(value).strip()
            if not value:
                continue
            if not key:
                unnamed.append(value)
            elif key not in merged:
                merged[key] = value
    if unnamed:
        merged[DETAILS_UNNAMED_KEY] = "; ".join(([merged[DETAILS_UNNAMED_KEY]] if DETAILS_UNNAMED_KEY in merged
                                                 else []) + unnamed)
    return merged


def product_details_text(details: Dict[str, str]) -> str:
    """
    Build the text that represents product_details for indexing. Only the values are used, because they are
    self-describing ("Full Sleeve", "Round Neck", "Slim") while attribute names ("Fabric", "Pattern") occur in almost
    every product and carry no information. Exceptions: identifier attributes (DETAILS_EXCLUDED_KEYS) are skipped,
    and Yes/No attributes contribute the attribute name when the value is "Yes" ("Hooded") and nothing when "No".
    :param details: parsed product_details (see parse_product_details)
    :return: text to be passed to build_terms
    """
    parts = []
    for key, value in details.items():
        if key in DETAILS_EXCLUDED_KEYS:
            continue
        low = value.lower()
        if low in DETAILS_YES_VALUES:
            parts.append(key)
        elif low not in DETAILS_NO_VALUES:
            parts.append(value)
    return " ".join(parts)


def recover_brand(brand: Optional[str], url: Optional[str]) -> Optional[str]:
    """
    Recover the full brand name from the product URL. In the dataset the brand field is often truncated
    ("REEB", "Pu", "York") while the URL slug starts with the full brand ("reebok-full-sleeve-...").
    The brand tokens (connector words such as "and"/"by" ignored, since slugs drop them) must equal the first slug
    tokens, except the last one, which only has to be a prefix of the corresponding slug token.
    :param brand: brand as given in the dataset
    :param url: product URL
    :return: recovered brand in lower case (e.g. "reebok"), or None when it cannot be matched
    """
    # plain token split (no compound joining): in slugs every hyphen is a word separator ("u-s-polo-assn")
    brand_tokens = [t for t in _TOKEN_RE.findall(normalize_text(brand)) if t not in BRAND_CONNECTOR_WORDS]
    if not brand_tokens or not url:
        return None
    path = urlparse(url).path.strip("/").split("/")
    slug = _TOKEN_RE.findall(normalize_text(path[0])) if path else []
    n = len(brand_tokens)
    if len(slug) < n or slug[:n - 1] != brand_tokens[:-1] or not slug[n - 1].startswith(brand_tokens[-1]):
        return None
    return " ".join(slug[:n])


def parse_number(value) -> Optional[float]:
    """
    Parse a display number into a float: prices ("2,999" -> 2999.0), discounts ("69% off" -> 69.0) and ratings
    ("3.9" -> 3.9).
    :param value: raw value (string or number)
    :return: float, or None when missing or not parseable
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)
    match = _NUMBER_RE.search(str(value).replace(",", ""))
    return float(match.group()) if match else None


def parse_bool(value) -> bool:
    """
    Parse out_of_stock into a bool (it is already a bool in the dataset; strings are handled for safety).
    :param value: raw value
    :return: bool (missing -> False, i.e. in stock)
    """
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def clean_string(value) -> Optional[str]:
    """
    Strip a string field and map empty strings to None, so that "missing" has a single representation.
    :param value: raw value
    :return: stripped string, or None
    """
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def preprocess_document(raw: dict, merged_fields: Optional[List[str]] = None, recover_brands: bool = True,
                        **term_options) -> dict:
    """
    Pre-process one raw document. The result keeps every output field (pid unchanged, numeric fields parsed into
    numbers, product_details as a dict, empty strings as None), adds "brand_full" (see recover_brand), one list of
    terms per text field ("<field>_terms") and "all_terms", the concatenation of the terms of merged_fields.
    :param raw: raw document as loaded from the JSON file
    :param merged_fields: text fields concatenated into all_terms (default: all of TEXT_FIELDS)
    :param recover_brands: build brand_terms from the brand recovered from the URL (falls back to the brand field)
    :param term_options: keyword arguments passed on to build_terms (stemmer, keep_numbers, ...)
    :return: processed document as a dict
    """
    merged_fields = TEXT_FIELDS if merged_fields is None else merged_fields
    details = parse_product_details(raw.get("product_details"))
    doc = {
        "pid": raw["pid"],
        "title": clean_string(raw.get("title")),
        "description": clean_string(raw.get("description")),
        "brand": clean_string(raw.get("brand")),
        "category": clean_string(raw.get("category")),
        "sub_category": clean_string(raw.get("sub_category")),
        "product_details": details,
        "seller": clean_string(raw.get("seller")),
        "out_of_stock": parse_bool(raw.get("out_of_stock")),
        "selling_price": parse_number(raw.get("selling_price")),
        "discount": parse_number(raw.get("discount")),
        "actual_price": parse_number(raw.get("actual_price")),
        "average_rating": parse_number(raw.get("average_rating")),
        "url": clean_string(raw.get("url")),
    }
    # brand_full is always lower case: the recovered name, or the normalized brand field when it cannot be recovered
    doc["brand_full"] = ((recover_brand(doc["brand"], doc["url"]) if recover_brands else None)
                         or " ".join(_TOKEN_RE.findall(normalize_text(doc["brand"]))) or None)
    field_text = {field: doc[field] for field in TEXT_FIELDS}
    field_text["brand"] = doc["brand_full"]
    field_text["product_details"] = product_details_text(details)
    for field in TEXT_FIELDS:
        doc[f"{field}_terms"] = build_terms(field_text[field], **term_options)
    doc["all_terms"] = [term for field in merged_fields for term in doc[f"{field}_terms"]]
    return doc


def preprocess_corpus(raw_docs: List[dict], merged_fields: Optional[List[str]] = None, recover_brands: bool = True,
                      **term_options) -> pd.DataFrame:
    """
    Pre-process every document of the corpus (see preprocess_document).
    :param raw_docs: list of raw documents (e.g. json.load of fashion_products_dataset.json)
    :param merged_fields: text fields concatenated into all_terms
    :param recover_brands: see preprocess_document
    :param term_options: keyword arguments passed on to build_terms
    :return: DataFrame with one row per document, indexed by pid (pid is also kept as a column)
    """
    rows = [preprocess_document(raw, merged_fields, recover_brands, **term_options) for raw in raw_docs]
    return pd.DataFrame(rows).set_index("pid", drop=False)


def preprocess_query(query: str, **term_options) -> List[str]:
    """
    Pre-process a query with the same pipeline as the documents.
    :param query: query text
    :param term_options: keyword arguments passed on to build_terms (must match the ones used for documents)
    :return: list of query terms
    """
    return build_terms(query, **term_options)


def save_processed_corpus(df: pd.DataFrame, path) -> None:
    """
    Save the processed corpus as JSON lines (one document per line), to be loaded in later parts.
    :param df: DataFrame returned by preprocess_corpus
    :param path: output file path
    """
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def load_processed_corpus(path) -> pd.DataFrame:
    """
    Load a corpus saved with save_processed_corpus.
    :param path: input file path
    :return: DataFrame indexed by pid
    """
    df = pd.read_json(path, orient="records", lines=True, dtype={"pid": str})
    return df.set_index("pid", drop=False)
