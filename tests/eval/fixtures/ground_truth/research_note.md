# Evaluation Note: Retrieval Quality

We compared three retrieval configurations on an internal benchmark of 500 queries. Scores are NDCG at 5, higher is better.

| Configuration | NDCG@5 | Latency p95 |
| --- | --- | --- |
| Baseline BM25 | 0.41 | 40 ms |
| Dense only | 0.58 | 90 ms |
| Hybrid rerank | 0.67 | 140 ms |

The hybrid configuration won on quality at an acceptable latency cost and is recommended for the next release.
