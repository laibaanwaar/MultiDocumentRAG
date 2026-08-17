from __future__ import annotations

from rag.intent_router import detect_legal_concepts, route_question


def test_detect_legal_concepts_finds_terrorism_from_direct_language() -> None:
    concepts = detect_legal_concepts(
        "What does terrorism mean under Section 6?"
    )

    assert "terrorism" in concepts


def test_section_6_terrorism_question_routes_to_ata_with_explicit_section() -> None:
    plan = route_question(
        "What does terrorism mean under Section 6?"
    )

    assert plan.question_type == "section_lookup"
    assert "terrorism" in plan.concepts
    assert plan.document_ids == ["ata_1997"]
    assert plan.section_number == "6"
    assert plan.provision_type == "section"
    assert "6" in plan.provision_numbers


def test_bare_terrorism_question_routes_to_ata_without_inventing_section() -> None:
    plan = route_question("What is terrorism?")

    assert "terrorism" in plan.concepts
    assert plan.document_ids == ["ata_1997"]
    assert plan.section_number is None
    assert "6" not in plan.provision_numbers
    assert plan.provision_type is None or plan.provision_type == "section"


def test_explicit_section_6_anti_terrorism_act_question_remains_ata_scoped() -> None:
    plan = route_question(
        "Explain Section 6 of the Anti-Terrorism Act"
    )

    assert plan.question_type == "section_lookup"
    assert plan.document_ids == ["ata_1997"]
    assert plan.section_number == "6"
    assert plan.provision_type == "section"


def test_bare_section_8_is_not_globally_hard_coded_to_ata() -> None:
    plan = route_question("What does Section 8 cover?")

    assert plan.section_number == "8"
    assert "ata_1997" not in plan.document_ids


def test_explicit_section_8_of_anti_terrorism_act_routes_to_ata() -> None:
    plan = route_question("Explain Section 8 of the Anti-Terrorism Act")

    assert plan.question_type == "section_lookup"
    assert plan.document_ids == ["ata_1997"]
    assert plan.section_number == "8"
    assert plan.provision_type == "section"


def test_unrelated_violent_conduct_does_not_trigger_terrorism_concept() -> None:
    concepts = detect_legal_concepts("violent conduct")

    assert "terrorism" not in concepts


def test_theft_routing_remains_unchanged() -> None:
    plan = route_question(
        "Ali dishonestly took a mobile phone without the owner's consent."
    )

    assert plan.question_type == "fact_scenario"
    assert plan.concepts == ["theft"]
    assert plan.document_ids == ["ppc_1860"]
    assert set(plan.provision_numbers) == {"378", "379"}
