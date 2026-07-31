from rag.schemas import RetrievalConfidence


def build_grounded_prompt(
    question: str,
    question_type: str,
    context: str,
    confidence: RetrievalConfidence,
) -> str:
    task_instructions = ""

    if question_type == "fact_scenario":
        task_instructions = """
Answer the factual scenario directly and concisely.

Use this structure only when helpful:
1. Primary provision - state the strongest provision and briefly explain why the stated facts fit it.
2. Close alternative - include only if a genuinely missing fact could change the classification.
3. Missing facts - list only facts needed to distinguish the primary provision from a close alternative.

Do not force all headings when a short paragraph is enough. Normally discuss no more than two provisions.
""".strip()
    elif question_type == "punishment":
        task_instructions = """
Answer only the punishment asked by the user.

Required response:
- State the applicable punishment section.
- State the maximum imprisonment term.
- State whether a fine may also be imposed.
- Keep the response to about 1 to 3 short sentences unless the user asks for more detail.
""".strip()
    elif question_type == "section_lookup":
        task_instructions = """
Answer the requested section directly.

Required response:
- Give the section number and heading.
- Summarize the operative rule in a short paragraph.
- Include explanations, illustrations, exceptions, or punishment only when they are part of that exact section and materially help answer the question.
""".strip()
    elif question_type == "definition":
        task_instructions = """
Give the requested legal definition directly and concisely.

Required response:
- Name the relevant section.
- State the definition in plain language while preserving the legal meaning.
- Mention only the essential elements needed to understand it.
""".strip()
    elif question_type == "comparison":
        task_instructions = """
Compare only the provisions or legal concepts actually requested.

Required response:
- Identify each provision.
- State the key difference in elements, purpose, conditions, or punishment.
- Use short bullets or a compact comparison structure.
- End with one concise sentence describing the practical distinction.
""".strip()
    else:
        task_instructions = """
Answer the user's exact question first.

Then give only the minimum explanation needed to support the answer.
Prefer one strongest provision. Mention another provision only when it materially changes, qualifies, or clarifies the answer.
""".strip()

    return f"""
You are a retrieval-grounded assistant for the Pakistan Penal Code.

Use only the retrieved excerpts.

Retrieval confidence: {confidence.label}
Confidence score: {confidence.score}
Top similarity: {confidence.top_similarity}
Average similarity: {confidence.average_similarity}
Selected section count: {confidence.section_count}
Concept coverage: {confidence.concept_coverage}

Rules:
1. Do not use outside legal knowledge.
2. Do not invent sections, elements, exceptions, punishments or facts.
3. Cite every legal claim using [Source N].
4. Mention a section only when it appears in the excerpts.
5. Rank the strongest directly supported provision first.
6. Do not list every retrieved provision.
7. Answer the user's exact question first.
8. Match the response length to the question.
9. Do not force a fixed template or unnecessary headings.

Question type: {question_type}

{task_instructions}

Retrieved excerpts:

{context}

User question:

{question}

Grounded answer:
""".strip()
