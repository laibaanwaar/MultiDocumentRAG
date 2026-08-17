from __future__ import annotations


class RetrievalError(RuntimeError):
    """Base class for retrieval failures with safe diagnostic context."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        route: str | None = None,
        collection: str | None = None,
        category: str = "operational",
        original_exception: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.route = route
        self.collection = collection
        self.category = category
        self.original_exception = original_exception


class MandatoryRetrievalError(RetrievalError):
    """A mandatory retrieval step failed and must stop the pipeline."""


class OptionalRetrievalError(RetrievalError):
    """A supporting retrieval step failed but may be degraded safely."""
