import math
from array import array
from collections import Counter, defaultdict

import numpy as np

from myapp.search.preprocessing import build_terms


def create_index_tfidf(docs):
    """
    Implement the inverted index and compute tf, df and idf

    Argument:
    docs -- DataFrame with the processed corpus of Part 1 (we use the columns "pid" and "all_terms")

    Returns:
    index - the inverted index (implemented through a Python dictionary) containing terms as keys and the corresponding
    list of documents where these keys appears in (and the positions) as values.
    tf - normalized term frequency for each term in each document (same order as in index)
    df - number of documents each term appears in
    idf - inverse document frequency of each term
    """
    index = defaultdict(list)
    tf = defaultdict(list)  # term frequencies of terms in documents (documents in the same order as in the main index)
    df = defaultdict(int)  # document frequencies of terms in the corpus
    idf = defaultdict(float)
    num_documents = len(docs)

    for pid, terms in zip(docs["pid"], docs["all_terms"]):  # terms = title + description + ... + seller terms

        ## ===============================================================
        ## create the index for the current document and store it in current_page_index
        ## current_page_index ==> { 'term1': [current_doc, [list of positions]], ...}

        ## Example: if the current doc has pid "ABC" and its terms are ["slim", "jean", "blue", "jean"]:
        ## current_page_index ==> { 'slim': ["ABC", [0]], 'jean': ["ABC", [1, 3]], 'blue': ["ABC", [2]] }
        ## ===============================================================
        current_page_index = {}

        for position, term in enumerate(terms):
            if term in current_page_index:
                # the term is already in the index of this document: add the new position
                current_page_index[term][1].append(position)
            else:
                # first time we see the term in this document
                current_page_index[term] = [pid, array('I', [position])]  # 'I' indicates unsigned int

        # norm of the document vector (the same for all the terms of the document)
        norm = 0
        for term, posting in current_page_index.items():
            # posting ==> [current_doc, [list of positions]], so len(posting[1]) is the frequency of the term
            norm += len(posting[1]) ** 2
        norm = math.sqrt(norm)

        # calculate the tf (term frequency / norm) and the df of each term
        for term, posting in current_page_index.items():
            tf[term].append(np.round(len(posting[1]) / norm, 4))
            df[term] += 1  # one more document contains this term

        # merge the current document index with the main index
        for term, posting in current_page_index.items():
            index[term].append(posting)

    # compute the idf of each term (we can only do it at the end, when we know the df)
    for term in df:
        idf[term] = np.round(np.log(float(num_documents / df[term])), 4)

    return index, tf, df, idf


def search_and(query_terms, index):
    """
    Conjunctive (AND) search: the output is the set of documents that contain ALL the query terms.
    So, we get the documents of each query term and take the intersection of them.

    Argument:
    query_terms -- list of query terms (already preprocessed)
    index -- inverted index

    Returns:
    docs -- set of pids
    """
    if len(query_terms) == 0:
        return set()

    docs = None
    for term in query_terms:
        if term not in index:
            # no document contains this term, so no document contains all the terms
            return set()

        # store in term_docs the ids of the docs that contain "term"
        term_docs = set([posting[0] for posting in index[term]])

        if docs is None:
            docs = term_docs  # first term
        else:
            docs = docs & term_docs  # docs = docs intersection term_docs
    return docs


def rank_documents(query_terms, docs, index, idf, tf):
    """
    Perform the ranking of the results of a search based on the tf-idf weights

    Argument:
    query_terms -- list of query terms
    docs -- set of documents (pids) to rank, matching the query
    index -- inverted index data structure
    idf -- inverse document frequencies
    tf -- term frequencies

    Returns:
    doc_scores -- list of [score, pid], sorted from the highest score to the lowest
    """
    # frequency of each term in the query, e.g. Counter(["jean", "blue", "jean"]) --> {'jean': 2, 'blue': 1}
    query_terms_count = Counter(query_terms)
    query_norm = np.linalg.norm(list(query_terms_count.values()))

    # each term must appear only once in the vectors, so we keep the unique query terms
    unique_terms = list(query_terms_count.keys())

    # We are only interested in the elements of the doc vectors for the query terms
    # (the other elements would become 0 when multiplied with the query vector)
    doc_vectors = defaultdict(lambda: [0] * len(unique_terms))
    query_vector = [0] * len(unique_terms)

    for term_index, term in enumerate(unique_terms):  # term_index is the position of the term in the vectors
        if term not in index:
            continue

        # tf-idf of the term in the query (tf normalized as done with the documents)
        query_vector[term_index] = query_terms_count[term] / query_norm * idf[term]

        # tf-idf of the term in each matching document
        for doc_index, (doc, postings) in enumerate(index[term]):
            # tf[term][doc_index] is the tf of "term" in the document index[term][doc_index]
            if doc in docs:
                doc_vectors[doc][term_index] = tf[term][doc_index] * idf[term]

    # score of each document = dot product of the document vector and the query vector
    doc_scores = [[np.dot(cur_doc_vector, query_vector), doc] for doc, cur_doc_vector in doc_vectors.items()]
    doc_scores.sort(reverse=True)
    return doc_scores


def search_in_corpus(query, index, tf, idf, top_k=None):
    """
    Search a query: preprocess it, find the documents with all its terms and rank them with tf-idf.

    Argument:
    query -- query text, e.g. "men slim jeans blue"
    index, tf, idf -- output of create_index_tfidf
    top_k -- number of results to return (None = all of them)

    Returns:
    doc_scores -- list of [score, pid], best first
    """
    query_terms = build_terms(query)  # same preprocessing as the documents
    docs = search_and(query_terms, index)
    doc_scores = rank_documents(query_terms, docs, index, idf, tf)
    if top_k is not None:
        doc_scores = doc_scores[:top_k]
    return doc_scores
