import math

## All the metrics get:
## ranked_pids -- list of pids in the order returned by the search engine (best first)
## relevant_pids -- set of the pids that are relevant for the query
## k -- number of documents to consider (the top k)


def precision_at_k(ranked_pids, relevant_pids, k):
    """
    Precision@K: fraction of the top k documents that are relevant

    Returns:
    precision @k : float
    """
    top_k = ranked_pids[:k]
    relevant_found = 0
    for pid in top_k:
        if pid in relevant_pids:
            relevant_found += 1
    return relevant_found / k


def recall_at_k(ranked_pids, relevant_pids, k):
    """
    Recall@K: fraction of all the relevant documents that are in the top k

    Returns:
    recall @k : float
    """
    if len(relevant_pids) == 0:  # no relevant documents for this query
        return 0.0
    top_k = ranked_pids[:k]
    relevant_found = 0
    for pid in top_k:
        if pid in relevant_pids:
            relevant_found += 1
    return relevant_found / len(relevant_pids)


def f1_at_k(ranked_pids, relevant_pids, k):
    """
    F1-score@K: harmonic mean of Precision@K and Recall@K

    Returns:
    f1 @k : float
    """
    precision = precision_at_k(ranked_pids, relevant_pids, k)
    recall = recall_at_k(ranked_pids, relevant_pids, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def avg_precision_at_k(ranked_pids, relevant_pids, k):
    """
    Average Precision@K: we compute the precision at each position (up to k) where there is a
    relevant document, add them up, and divide by min(number of relevant documents, k)

    Returns:
    average precision @k : float
    """
    if len(relevant_pids) == 0:
        return 0.0

    relevant_found = 0
    prec_at_i_sum = 0.0
    for rank, pid in enumerate(ranked_pids[:k], start=1):  # rank starts at 1
        if pid in relevant_pids:
            relevant_found += 1
            prec_at_i_sum += relevant_found / rank  # precision at this position
    return prec_at_i_sum / min(len(relevant_pids), k)


def rr_at_k(ranked_pids, relevant_pids, k):
    """
    Reciprocal Rank@K: 1 / position of the first relevant document in the top k

    Returns:
    reciprocal rank for the current query (0 if there is no relevant document in the top k)
    """
    for rank, pid in enumerate(ranked_pids[:k], start=1):
        if pid in relevant_pids:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked_pids, relevant_pids, k):
    """
    NDCG@K with binary relevance (1 relevant, 0 not relevant):
    DCG of our ranking divided by the DCG of the ideal ranking (all the relevant documents first)

    Returns:
    ndcg @k : float
    """
    # DCG of our ranking: gain 1 for each relevant document, divided by log2(position + 1)
    dcg = 0.0
    for rank, pid in enumerate(ranked_pids[:k], start=1):
        if pid in relevant_pids:
            dcg += 1 / math.log2(rank + 1)

    # DCG of the ideal ranking: the first min(number of relevant, k) positions are relevant
    ideal_relevant = min(len(relevant_pids), k)
    idcg = 0.0
    for rank in range(1, ideal_relevant + 1):
        idcg += 1 / math.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_average_precision(query_rankings, query_relevance, k):
    """
    MAP@K: mean of the Average Precision@K of all the queries

    Argument:
    query_rankings -- dictionary: query_id -> ranked_pids
    query_relevance -- dictionary: query_id -> relevant_pids
    k -- number of documents to consider

    Returns:
    mean average precision @k : float
    """
    ap_list = []
    for query_id in query_rankings:  # loop over all the queries
        ap_list.append(avg_precision_at_k(query_rankings[query_id], query_relevance[query_id], k))
    return sum(ap_list) / len(ap_list)


def mean_reciprocal_rank(query_rankings, query_relevance, k):
    """
    MRR@K: mean of the Reciprocal Rank@K of all the queries (same arguments as mean_average_precision)

    Returns:
    mean reciprocal rank @k : float
    """
    rr_list = []
    for query_id in query_rankings:  # loop over all the queries
        rr_list.append(rr_at_k(query_rankings[query_id], query_relevance[query_id], k))
    return sum(rr_list) / len(rr_list)


def evaluate_query(ranked_pids, relevant_pids, k):
    """
    Compute all the metrics @k of one query

    Returns:
    result -- dictionary: metric name -> value
    """
    result = {}
    result["Precision"] = precision_at_k(ranked_pids, relevant_pids, k)
    result["Recall"] = recall_at_k(ranked_pids, relevant_pids, k)
    result["F1"] = f1_at_k(ranked_pids, relevant_pids, k)
    result["AP"] = avg_precision_at_k(ranked_pids, relevant_pids, k)
    result["RR"] = rr_at_k(ranked_pids, relevant_pids, k)
    result["NDCG"] = ndcg_at_k(ranked_pids, relevant_pids, k)
    return result
