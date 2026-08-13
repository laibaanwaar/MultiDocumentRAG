from __future__ import annotations

import re
from typing import Any

_CITATION_PREFIX_PATTERN = (
    r"(?:section|sections|sec\.?|secs\.?|s\.?|"
    r"article|articles|art\.?)"
)

_PROVISION_NUMBER_PATTERN = re.compile(
    rf"^\s*"
    rf"(?:{_CITATION_PREFIX_PATTERN}\s*)?"
    r"(?P<digits>\d+)"
    r"(?:(?:\s*-\s*(?P<hyphen_suffix>[A-Za-z]{1,3}))|"
    r"(?P<suffix>[A-Za-z]{1,3}))?"
    r"(?P<children>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*)"
    r"\s*"
    r"[.,;:]*\s*$",
    flags=re.IGNORECASE,
)


def canonicalize_provision_number(
    value: str | int | None,
) -> str | None:
    """
    Canonicalize a legal provision number to an uppercase base identifier.

    The result strips recognized citation prefixes, removes whitespace,
    normalizes the suffix hyphen away, and discards any child path.
    """

    if value is None:
        return None

    normalized = str(value).strip()

    if not normalized:
        return None

    match = _PROVISION_NUMBER_PATTERN.match(
        normalized
    )

    if not match:
        return None

    digits = match.group("digits")
    suffix = (
        match.group("hyphen_suffix")
        or match.group("suffix")
        or ""
    )

    return f"{digits}{suffix.upper()}"


def canonicalize_provision_number_or_none(
    value: Any,
) -> str | None:
    """Backward-compatible helper for loosely typed call sites."""

    return canonicalize_provision_number(
        value
    )
