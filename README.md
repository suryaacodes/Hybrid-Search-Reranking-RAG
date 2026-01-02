# Hybrid-Search-Reranking-RAG 

This repo is a small RAG system you can run locally.



## Goal

Answer internal-style policy/runbook questions with citations.

## Constraints

- Uses synthetic docs (no private company material), but the corpus is intentionally *messy*:
  duplicates, outdated clauses, inconsistent headings, and a mix of one-liners + long policies.
- Local-first: run with FastAPI + persisted index on disk
- Prefer inspectable retrieval over “magic”: you can see top chunks + signals

## What’s implemented

- `/ingest`: build an index from `data/sample_docs.json`
- `/search`: hybrid retrieval (BM25 + dense) + cross-encoder reranking
- `/ask`: retrieval + prompt payload (answering is a separate step you can swap)

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

```bash
# Build the index
curl -X POST http://127.0.0.1:8000/ingest
```

```bash
# Inspect retrieval
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the airport rideshare reimbursement limit?", "k": 5}'
```

```bash
# Ask (prompt payload returned with citations)
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Are alcoholic drinks reimbursable?", "k": 5}'
```

## Engineering decisions

These are the “knobs” I expect to tune per corpus.

- **Chunk size (~650 words) + overlap (120 words)**  
  I want a single chunk to usually contain a rule + its exception (deadline + approval ladder),
  but not so large that ranking gets diluted. Overlap is high because policies often have
  definitions at the top and constraints at the bottom.

- **BM25 weight (~0.35)**  
  Dense retrieval handles paraphrases. BM25 rescues exact tokens (numbers, acronyms, “30 days”, “SEV1”, “#it-helpdesk”).
  I keep the weight modest because BM25 can overfit to short/outdated pages.

- **Reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**  
  This is a latency/quality trade. It’s fast enough for a portfolio repo and still demonstrates
  the “precision step”. Larger rerankers can do better but cost more and are harder to run locally.

## One real failure example (debug notes)

**Query:** “What is the airport rideshare reimbursement limit?”

**What happened (failed run):**
- Top retrieval hit was an *outdated* page: `policy/travel/local_limits_MI.md` (says **$60**).
- The newer travel quick-ref (says **$75**) was sometimes missing from the candidate pool.

**Why retrieval failed:**
- The outdated doc is short and keyword-dense: “airport / limit / $60 / trip”.
- BM25 likes exact keyword overlap, and the short doc concentrates those tokens.
- Dense retrieval also pulled it because the semantic neighborhood is similar.

**Did reranking help?**
- **Not at first.** Reranking can only reorder what it sees.
- When the correct travel chunk wasn’t in the candidate pool, reranking could not recover.

**What I changed:**
- Increased `rerank_k` (candidate pool for reranking) so the travel quick-ref stays in scope.
- Nudged `bm25_weight` up slightly (but kept it <0.5) and increased `bm25_k` to keep exact-token docs *and* the longer policy chunk.
- Added `updated_at` + `source` to every doc so conflicts are obvious in the API response.

You can see more examples in `data/failure_examples.json`.

## Failure examples (5)

These are real “retrieval debugging” style notes:

1) Late receipts (“after 30 days”) → missed the approval ladder → overlap ↑, rerank_k ↑  
2) Boundary question (“6.5 hour flight”) → rule split across chunks → return more hits + add eval case  
3) Contradictory policy (“alcohol with client present”) → outdated doc surfaced → expose freshness + document limitations  
4) Split facts (“approver + SLA”) → facts in two sections → return more hits; next is merge/compose  
5) Escalation (“blocked SSO login”) → generic docs matched → bm25_k ↑ for exact `#it-helpdesk`

## Trade-offs & Limitations

- **Rerankers add latency/cost.** Cross-encoders are the first thing you turn off in production when QPS spikes.
  This repo keeps reranking explicit so you can measure it and decide.
- **Chunking is sensitive.** Small changes in chunk size/overlap can move a key sentence out of the top-k.
  I treat chunking as a tunable component, not a constant.
- **No embedding fine-tuning (yet).** I’m using off-the-shelf models to keep the repo runnable locally.
  Next step would be domain adaptation (or at least a better embedding model) once I have a labeled eval set.


