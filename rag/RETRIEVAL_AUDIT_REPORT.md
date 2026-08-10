# Legal RAG Retrieval Audit Report

**Audit date:** 10 August 2026
**Reviewed component:** `retriever.py`
**Audit type:** Static functional and reliability review
**Primary concern:** Broad legal questions perform reasonably, while detailed questions produce incomplete or inaccurate answers.

## 1. Executive summary

The reviewed retriever is reasonably suited to discovering broadly relevant legal provisions, but it does not reliably retrieve precise clauses, exceptions, provisos, definitions, or all chunks belonging to a cited provision.

The central design gap is that an “exact” Section or Article request is still executed as a top-*k* semantic vector search inside a metadata filter. Consequently, only the chunks whose embeddings are closest to the question are returned. If the answer is located in a less semantically similar subclause, it can be omitted even though the correct provision was identified.

This problem is amplified by an early-return rule: after finding any exact chunk for a Section, Article, or fact scenario, the function stops before semantic and neighboring retrieval can supply missing material. The implementation also has no explicit subsection/clause model, no lexical search, no final deduplication or reranking of general candidates, and limited retrieval observability.

### Overall assessment

**Risk rating: High for detailed legal questions.**

The main risk is retrieval incompleteness rather than the language model alone. A model cannot reliably answer a detailed question when the relevant clause was never supplied in its context. Prompt changes or switching models may improve presentation, but they will not correct missing evidence.

### Highest-priority actions

1. Fetch cited provisions deterministically by metadata instead of using top-*k* similarity as the only fetch mechanism.
2. Remove the premature exact-result short circuit and merge exact, semantic, and supporting results.
3. Represent and route subsection/clause references explicitly.
4. Add hybrid lexical/vector retrieval, deduplication, and reranking.
5. Add retrieval traces and an evaluation set based on real detailed questions.

## 2. Scope and limitations

Only `retriever.py` was available for review. The audit therefore covers retrieval logic visible in that module.

The following components were not available and must be reviewed before production sign-off:

- document ingestion and chunking;
- OCR and text-cleaning quality;
- Qdrant collection configuration and payload indexes;
- embedding model and distance metric;
- question classification and query expansion;
- legal concept registry contents;
- context assembly and token budgeting;
- answer-generation prompts and citation validation;
- production logs and real failing query traces.

Findings labelled **Confirmed** are directly evident in the reviewed code. Findings labelled **Likely** require the missing components or runtime data to confirm.

## 3. Current retrieval flow

The main function, `fetch_candidates`, currently performs the following:

1. Normalizes supplied document and provision identifiers.
2. If a Section or Article was detected, performs filtered vector searches for that provision.
3. If any exact result is found for a lookup or fact-scenario question, immediately returns those results.
4. Otherwise performs dense vector searches for each generated query.
5. Optionally searches numerically neighboring Sections suggested by detected concepts.
6. Returns semantic candidates and exact documents to downstream processing.

This flow favors topical relevance. It does not guarantee that all legally material text belonging to a cited provision is present.

## 4. Detailed findings and remediation

### F-01 — “Exact” provision retrieval is still top-*k* semantic retrieval

**Severity:** Critical
**Confidence:** Confirmed
**Evidence:** `retrieve_exact_provision_documents`, approximately lines 834–843

The code applies a provision metadata filter and then calls `similarity_search_with_score` using a bounded value of *k*. This restricts the search to the correct provision but does not fetch the provision completely.

**Impact**

- A long Section split into several chunks may be only partially returned.
- Exceptions, explanations, provisos, illustrations, penalties, and definitions can be missed.
- A broad question may match a heading or opening chunk, while a detailed question needs a later chunk.
- Increasing `top_k` may reduce the symptom but does not guarantee completeness.

**Required fix**

Implement a deterministic metadata retrieval path for explicit citations:

- filter by `document_id`, `provision_type`, and canonical base provision number;
- use Qdrant payload scrolling or an equivalent non-vector lookup to obtain every usable chunk for that provision;
- sort chunks by an explicit order field such as `provision_chunk_index`, falling back to page and document chunk order;
- if a subsection was requested, prioritize its chunks but retain enough parent provision context to interpret it;
- apply an explicit context-budget policy if a provision is exceptionally long; do not silently truncate based only on semantic score.

**Acceptance criteria**

- Requesting a known provision returns all of its indexed body chunks in source order.
- A test answer located in the final chunk of a long provision remains retrievable regardless of embedding similarity.
- Returned chunks cannot come from a different document or provision.

### F-02 — Premature return suppresses complementary retrieval

**Severity:** Critical
**Confidence:** Confirmed
**Evidence:** `fetch_candidates`, approximately lines 1088–1100

For `section_lookup`, `article_lookup`, and `fact_scenario`, the presence of any exact document causes an immediate return with an empty semantic-candidate list.

**Impact**

- One partial exact hit is treated as sufficient evidence.
- Definitions located elsewhere, cross-referenced provisions, and applicable related Sections are excluded.
- Fact scenarios—which are typically multi-provision questions—are especially likely to receive incomplete context.

**Required fix**

Do not treat “at least one exact chunk found” as completion. Merge retrieval channels:

- exact citation results receive mandatory/highest priority;
- semantic and lexical searches add definitions and related provisions;
- explicit cross-references and concept-supported provisions are added with bounded priority;
- deduplicate and rerank the combined set before context assembly.

An early return is safe only after deterministic retrieval has confirmed that the requested material is complete and the question type genuinely requires no supporting sources.

**Acceptance criteria**

- Fact-scenario tests include exact and supporting evidence where relevant.
- A partial exact hit never prevents retrieval of the clause containing the expected answer.
- The retrieval trace records why each supporting provision was included.

### F-03 — No explicit subsection, clause, paragraph, or proviso routing

**Severity:** High
**Confidence:** Confirmed in this module; ingestion support is unknown
**Evidence:** Filters use provision, Section, Article, and document fields only

Detailed legal references commonly include structures such as `9(2)(b)`, an explanation, proviso, schedule item, or rule paragraph. The reviewed interface has no dedicated fields for these structures.

**Impact**

- The system can identify the parent Section but not reliably select the requested child clause.
- A compound citation may fail exact equality against a base provision number.
- Different child clauses of the same Section compete using embedding similarity alone.

**Required fix**

Introduce a canonical legal-reference object, for example:

```text
document_id: act_x
provision_type: section
base_number: 9A
subsection_path: ["2", "b"]
component_type: clause
```

Store corresponding metadata during ingestion. The query parser should extract both the base number and child path. Retrieval should use exact child-path matching first, then back off to the base provision when the child is absent or uncertain.

**Acceptance criteria**

- `Section 9`, `s. 9`, and equivalent forms resolve to one canonical base identifier.
- `9(1)` and `9(2)` retrieve distinguishable child chunks.
- Failure to find an exact child path triggers a logged parent-level fallback rather than an empty or unrelated result.

### F-04 — Provision-number normalization is insufficient

**Severity:** High
**Confidence:** Confirmed
**Evidence:** `normalize_provision_number`, approximately lines 19–29

Normalization only strips surrounding whitespace and converts text to uppercase. It does not parse prefixes, punctuation, suffix conventions, or child references.

**Impact**

Equivalent forms such as `Section 9-A`, `s 9A`, and collection-specific values may not match. A compound reference such as `9(2)` may be compared with metadata containing only `9`.

**Required fix**

- Parse, do not merely uppercase, legal references.
- Remove recognized labels such as `section`, `sec.`, `article`, and `art.`.
- Define one canonical suffix and punctuation policy.
- Separate the base number from the subsection path.
- Preserve the original citation for display and audit logging.
- Apply the same canonicalizer during both ingestion and querying.

**Acceptance criteria**

- A table-driven test suite maps all supported citation variants to the same canonical identifier.
- Ingestion-time and query-time canonicalization produce identical outputs.

### F-05 — Dense vector search is the only general retrieval method

**Severity:** High
**Confidence:** Confirmed
**Evidence:** `AdaptiveRetriever` calls only similarity-search methods

Embeddings are good at topical similarity but weaker for exact phrases, identifiers, rare terminology, numerical thresholds, dates, and legally significant negation.

**Impact**

- Broad conceptual questions perform better than precise textual questions.
- Semantically similar but legally different clauses may outrank the correct text.
- Exact terms quoted by the user are not given special weight.

**Required fix**

Adopt hybrid retrieval:

1. deterministic metadata retrieval for explicit citations;
2. lexical/BM25 or full-text retrieval for exact language and identifiers;
3. dense retrieval for conceptual similarity;
4. rank fusion, followed by a legal-domain reranker if available;
5. source-aware context assembly.

Reciprocal Rank Fusion is a suitable initial method because lexical and vector scores are not necessarily calibrated to the same scale.

**Acceptance criteria**

- Exact-phrase and rare-term tests retrieve the expected clause within the final context.
- Hybrid retrieval outperforms dense-only retrieval on a held-out detailed-query set.
- Negation and numerical-condition tests are included in evaluation.

### F-06 — General and neighbor candidates are not deduplicated or globally reranked

**Severity:** High
**Confidence:** Confirmed
**Evidence:** `fetch_candidates`, approximately lines 1110–1223

Candidates from each query and neighbor search are appended independently. Deduplication exists for exact results, but not for the final semantic candidate collection.

**Impact**

- Repeated chunks can consume the downstream context budget.
- Results are grouped by query execution rather than final relevance.
- Broadly similar chunks can crowd out one precise but lower-ranked chunk.

**Required fix**

- Deduplicate all candidates using stable `chunk_id` or the existing document-identity function.
- Preserve provenance from every retrieval route that found a chunk.
- Fuse ranks across queries and retrieval methods.
- Rerank the combined pool against the original user question.
- enforce diversity by provision/document where useful, while pinning explicit citations.

**Acceptance criteria**

- No duplicate chunk appears in assembled context.
- Final ordering is independent of query-loop insertion order.
- Retrieval provenance remains inspectable after rank fusion.

### F-07 — Concept-neighbor loop creates a cross-product of concepts and Sections

**Severity:** Medium to High
**Confidence:** Confirmed
**Evidence:** `fetch_candidates`, approximately lines 1148–1190

The code first unions preferred Sections from all detected concepts. It then loops over every concept and searches every preferred Section using the current concept’s document hints.

**Impact**

- A Section belonging to concept A can be searched against concept B’s documents.
- Duplicate searches and duplicate candidates are generated.
- Irrelevant neighbor material can dilute precise evidence.

**Required fix**

Keep each concept associated with its own preferred Sections and document IDs. Build unique route tuples such as `(document_id, provision_type, provision_number)` and execute each route once.

**Acceptance criteria**

- Two concepts with disjoint preferred documents never exchange Section hints.
- Identical route tuples execute only once.
- Unit tests cover multiple detected concepts.

### F-08 — Exceptions are swallowed and retrieval can fail silently

**Severity:** High
**Confidence:** Confirmed
**Evidence:** broad `except Exception` handling in exact and neighbor retrieval

Filter schema errors, missing payload indexes, client failures, and incompatible metadata can all be converted into empty results without a diagnostic signal.

**Impact**

- Operational failures appear to users as inaccurate answers.
- Developers cannot distinguish “no legal material exists” from “retrieval failed.”
- Regressions can remain undetected.

**Required fix**

- Catch only expected, specific exceptions.
- Emit structured logs with query ID, route, filter, collection, latency, and error category.
- Return an explicit retrieval-status object or raise failures that make evidence incomplete.
- Prevent answer generation—or clearly degrade with a warning—when mandatory exact retrieval fails operationally.

Do not log sensitive user facts or full document content unless the data-governance policy permits it.

**Acceptance criteria**

- A deliberately invalid filter generates an observable error rather than an unexplained empty list.
- Monitoring distinguishes zero matches, threshold rejection, timeout, and system error.

### F-09 — Score fallback conflicts with positive relevance thresholds

**Severity:** Medium
**Confidence:** Confirmed, conditional on fallback use
**Evidence:** `search_with_scores`, approximately lines 716–730; threshold checks near lines 1130 and 1202

If scored search is unavailable, every fallback result receives a score of `0.0`. If `min_relevance_score` is positive, all such results are discarded.

**Required fix**

- Represent an unavailable score as `None`, not as a real zero score.
- Use rank-based fallback logic when scores are unavailable.
- Validate at startup that the configured vector store supports the required scored API.

**Acceptance criteria**

- API fallback does not silently turn a successful search into zero accepted candidates.
- Thresholds are calibrated for the actual embedding model and distance metric.

### F-10 — Neighbor expansion only supports purely numeric provisions

**Severity:** Medium
**Confidence:** Confirmed
**Evidence:** `build_neighbor_provision_numbers`, approximately lines 612–646

Alphanumeric provisions such as `9A` are returned alone and therefore never receive neighbor expansion. More importantly, numerical adjacency is not always equivalent to legal relevance.

**Required fix**

- Prefer source-order adjacency recorded at ingestion over integer arithmetic.
- Store previous/next provision IDs or a stable provision ordinal.
- Treat cross-references and shared definitions as stronger relationships than simple numerical proximity.

**Acceptance criteria**

- Alphanumeric and inserted provisions have correct previous/next relationships.
- Neighbor expansion cannot cross into a different document unintentionally.

### F-11 — Metadata quality can cause valid chunks to be rejected

**Severity:** Medium
**Confidence:** Likely; rejection behavior is confirmed, data quality is unknown
**Evidence:** `is_usable_document` and `document_matches_route`, approximately lines 502–580

The retriever strictly rejects chunks based on metadata such as `heading_only_chunk`, body-presence flags, document ID, provision type, and provision number. This is appropriate only if ingestion applies these fields consistently.

**Required fix**

- Define and version a payload schema.
- Validate every chunk before insertion.
- Run collection audits for missing, null, inconsistent, or wrongly typed fields.
- Add migration tooling when metadata conventions change.
- Maintain collection-level counts by document, provision, and rejection reason.

**Acceptance criteria**

- Schema-validation failure prevents malformed chunks from entering the production collection.
- A collection audit reports 100% coverage for mandatory routing and ordering fields.

### F-12 — No retrieval observability or regression evaluation is visible

**Severity:** High
**Confidence:** Likely at project level; absent from reviewed module

Without retrieval-level evaluation, teams tend to assess only whether the final answer sounds plausible. That cannot reveal whether the correct evidence was retrieved.

**Required fix**

For each request, record a safe structured trace containing:

- normalized question type and extracted citations;
- generated retrieval queries;
- filters and routing decisions;
- candidate chunk IDs, ranks, and scores;
- rejection reasons;
- final context chunk IDs and token counts;
- cited sources in the generated answer;
- latency and failures for each retrieval channel.

Create a versioned evaluation set with real broad and detailed legal questions. Evaluate retrieval separately from answer generation.

Recommended retrieval metrics include:

- Recall@k;
- Mean Reciprocal Rank;
- provision and clause hit rate;
- exact-citation completeness;
- duplicate-context rate;
- unsupported-answer/citation rate;
- failure rate and latency by retrieval route.

**Acceptance criteria**

- Every test question has expected supporting chunk or provision IDs.
- CI fails when detailed-query recall falls below the agreed baseline.
- Production debugging can show whether a wrong answer came from retrieval, context assembly, or generation.

## 5. Recommended target design

```text
User question
    |
    +-- Reference parser -> canonical document/provision/subsection route
    |
    +-- Question classifier with confidence and fallback
    |
    +-- Deterministic metadata fetch (mandatory cited material)
    +-- Lexical retrieval (exact terms, numbers, quotations)
    +-- Dense retrieval (conceptual relevance)
    +-- Cross-reference/definition expansion (bounded)
                         |
                    Deduplicate
                         |
                     Rank fusion
                         |
                       Rerank
                         |
          Context assembly with source order and budget
                         |
            Answer generation constrained to evidence
                         |
               Citation/entailment validation
```

Important policy: an explicitly cited provision should be treated as mandatory evidence, while semantic retrieval should supplement it rather than replace it.

## 6. Phased remediation plan

### Phase 1 — Immediate reliability corrections

**Indicative effort:** 1–3 engineering days, excluding testing and deployment

- Remove or narrow the premature exact-result return.
- Fix the concept/Section cross-product.
- Deduplicate the final candidate pool.
- Replace broad exception swallowing with structured errors and logs.
- Correct score-unavailable handling.
- Add unit tests for these branches.

**Expected result:** Less missing context, less duplicated/noisy context, and diagnosable failures.

### Phase 2 — Deterministic detailed retrieval

**Indicative effort:** 3–7 engineering days, dependent on the existing schema

- Add canonical reference parsing.
- Add subsection/clause and source-order metadata during ingestion.
- Fetch explicit provisions by payload rather than only vector similarity.
- Preserve provision order during context assembly.
- Reindex existing documents if required.

**Expected result:** Material improvement for detailed Section, Article, and clause questions.

### Phase 3 — Hybrid retrieval and reranking

**Indicative effort:** 1–2 weeks including evaluation

- Add lexical retrieval.
- Fuse exact, lexical, and dense candidate ranks.
- Add a reranking stage.
- Introduce bounded cross-reference and definition expansion.
- Calibrate retrieval depth and thresholds using evaluation data.

**Expected result:** Improved accuracy for exact wording, uncommon terminology, numerical conditions, and multi-provision scenarios.

### Phase 4 — Quality gate and production monitoring

**Indicative effort:** Ongoing

- Build the detailed-query gold set.
- Add retrieval regression tests to CI.
- Add dashboards for retrieval failures and detailed-query recall.
- Sample and review unsupported or low-confidence answers.
- Version embeddings, chunking rules, schemas, and evaluation results.

## 7. Required test matrix

At minimum, testing should include:

| Test category | Example intent | Expected behavior |
|---|---|---|
| Broad concept | “What are the main provisions governing X?” | Diverse, relevant primary provisions |
| Exact Section | “What does Section 9 provide?” | Complete Section in source order |
| Subsection | “What condition appears in Section 9(2)(b)?” | Exact child clause plus needed parent context |
| Exception/proviso | “What is the exception to the general rule?” | Proviso/exception chunk included |
| Definition dependency | “Does X include Y?” | Definition provision and operative provision |
| Cross-reference | “How does Section 9 apply subject to Section 12?” | Both provisions retrieved |
| Fact scenario | Multi-fact application question | Multiple applicable provisions; no premature stop |
| Exact phrase | Quoted statutory wording | Lexical route finds the precise clause |
| Negation | “When is X not permitted?” | Clause containing negation outranks general rule |
| Numerical condition | Threshold, time limit, age, or amount | Exact numerical clause included |
| Alphanumeric provision | Section `9A` or equivalent | Canonical match and correct adjacency |
| Ambiguous document | Same Section number in multiple laws | Correct document or explicit clarification |
| Missing citation | Nonexistent provision | Honest no-match result, not a guessed answer |
| Backend failure | Qdrant/filter error | Observable controlled failure |

## 8. Production acceptance criteria

Before declaring the issue resolved:

1. The gold evaluation set must include representative real failures supplied by users.
2. Explicit-citation completeness should be effectively 100% for correctly ingested material.
3. Detailed-query Recall@k and clause hit rate must meet an agreed target and improve materially over the current baseline.
4. No duplicate chunks should consume final context.
5. Mandatory retrieval failures must be visible and must not produce an unqualified legal answer.
6. Generated answers must cite retrieved source text, and citation checks must confirm that the cited text supports the material claim.
7. Broad-query quality must not regress while detailed-query recall improves.

## 9. Additional files and data requested for full audit

To complete root-cause confirmation, request the following from the project owner:

- ingestion/chunking source code;
- a sample Qdrant payload for a long Section with several subclauses;
- Qdrant collection and payload-index configuration;
- embedding model name and version;
- values for `top_k`, neighbor radius, and relevance threshold;
- question classifier and query-expansion code;
- context assembly and answer prompt;
- five to ten failing detailed questions with expected sources;
- retrieved chunk IDs and scores for those failures;
- examples of successful broad questions for regression comparison.

## 10. Conclusion

The reported behavior is consistent with the reviewed implementation. Broad searches benefit from dense semantic similarity, while detailed legal questions require deterministic citation retrieval, clause-aware metadata, lexical matching, and complete context assembly.

The recommended first correction is not a larger model or a prompt rewrite. It is to guarantee that the relevant legal text—especially every requested provision and child clause—is retrieved, ordered, deduplicated, and supplied to the model. Hybrid search, reranking, observability, and regression evaluation should then be added to make that guarantee reliable in production.

---

**Audit qualification:** This report identifies functional and reliability gaps in the supplied retrieval module. It is not a security audit or a legal opinion. Final conclusions should be validated against the ingestion pipeline, vector-store contents, downstream answer-generation code, and representative production queries.
