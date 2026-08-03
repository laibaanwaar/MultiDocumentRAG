from rag.schemas import RetrievalConfidence


TASK_INSTRUCTIONS = {
    "fact_scenario": """
Answer the scenario using only the retrieved excerpts.
State the strongest applicable provision first, then include any other materially relevant provisions if they change the legal result.
Do not stop after the first relevant sentence when more relevant context exists.
""".strip(),

    "punishment": """
Answer the punishment completely but stay focused on the retrieved provision.
Include the law name, Section or Article number, heading, imprisonment limits, fine, forfeiture, and any material condition or exception.
Cover all punishment options present in the excerpt and do not stop after the first relevant sentence when more relevant context exists.
""".strip(),

    "section_lookup": """
Answer the lookup completely but stay focused on the retrieved provision.
Include the law name, Section or Article number, heading, all materially relevant clauses, explanations, exceptions, consequences, and any punishment where relevant.
Do not stop after the first relevant sentence when more relevant context exists.
""".strip(),

    "article_lookup": """
Answer the lookup completely but stay focused on the retrieved provision.
Include the law name, Article number, heading, all materially relevant clauses, explanations, exceptions, safeguards, remedies, and consequences.
Do not stop after the first relevant sentence when more relevant context exists.
""".strip(),

    "definition": """
State the relevant law and provision, then give the full definition in plain language while preserving its legal meaning and essential elements.
Include any materially relevant explanation or exception in the retrieved excerpt.
Do not stop after the first relevant sentence when more relevant context exists.
""".strip(),

    "constitutional_right": """
Identify the relevant constitutional Article and explain the right, safeguard,
scope, and limitation supported by the retrieved text.
""".strip(),

    "comparison": """
Compare only the requested laws, provisions, or concepts.
Identify each provision, explain the main difference, and include any materially relevant clause, exception, or consequence.
Do not stop after the first relevant sentence when more relevant context exists.
""".strip(),

    "general": """
Answer the user's question directly, then provide only the minimum supporting explanation.
Prefer the strongest relevant provision.
""".strip(),
}


def build_grounded_prompt(
    question: str,
    question_type: str,
    context: str,
    confidence: RetrievalConfidence,
) -> str:
    """Build a concise, citation-grounded prompt for Pakistan legal documents."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context.strip():
        raise ValueError("Retrieved context cannot be empty.")

    task = TASK_INSTRUCTIONS.get(
        question_type,
        TASK_INSTRUCTIONS["general"],
    )

    return f"""
You are a retrieval-grounded assistant for these Pakistan legal documents:
- Pakistan Penal Code, 1860
- Constitution of Pakistan, 1973
- Anti-Terrorism Act, 1997
- Anti-Money Laundering Act, 2010

Use only the retrieved excerpts. Do not rely on outside legal knowledge.

Rules:
1. Answer the exact question first.
2. Do not begin the answer with a standalone citation.
3. Place [Source N] after each paragraph or materially distinct legal claim.
4. Use citations exactly as [Source N] with no extra text inside the brackets.
5. Mention only laws and provisions present in the excerpts.
6. Do not invent sections, articles, facts, exceptions, punishments, or conclusions.
7. Distinguish Section references from Constitution Article references.
8. For comparisons, keep each law and provision clearly separated.
9. If the retrieved context is incomplete, clearly say that the complete provision was not retrieved.
10. Do not stop after the first relevant sentence when more relevant context exists.
11. Keep the answer focused on the requested provision and include all materially relevant details from the excerpt.
12. Do not mention retrieval scores unless directly relevant.

Question type: {question_type}
Retrieval confidence: {confidence.label}

Task:
{task}

Retrieved excerpts:
{context}

User question:
{question}

Grounded answer:
""".strip()
