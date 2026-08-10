from rest_framework import status
from rest_framework.exceptions import APIException


class RagAuthenticationRequiredError(APIException):
    status_code = status.HTTP_401_UNAUTHORIZED
    default_code = "authentication_required"

    def __init__(self):
        super().__init__(
            detail={
                "code": "AUTHENTICATION_REQUIRED",
                "message": "Authentication credentials were not provided.",
            }
        )


class RagUserInactiveError(APIException):
    status_code = status.HTTP_403_FORBIDDEN
    default_code = "user_inactive"

    def __init__(self):
        super().__init__(
            detail={
                "code": "USER_INACTIVE",
                "message": "This account is inactive.",
            }
        )


class RagSubscriptionRequiredError(APIException):
    status_code = status.HTTP_403_FORBIDDEN
    default_code = "subscription_required"

    def __init__(self):
        super().__init__(
            detail={
                "code": "SUBSCRIPTION_REQUIRED",
                "message": "An active subscription is required.",
            }
        )


class RagQueryLimitExceededError(APIException):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    default_code = "query_limit_exceeded"

    def __init__(self):
        super().__init__(
            detail={
                "code": "QUERY_LIMIT_EXCEEDED",
                "message": "Your plan query limit has been reached.",
            }
        )


class RagInvalidQuestionError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "invalid_question"

    def __init__(self, message: str = "The question is invalid."):
        super().__init__(
            detail={
                "code": "INVALID_QUESTION",
                "message": message,
            }
        )


class RagInvalidRequestError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "invalid_request"

    def __init__(self, message: str = "The request payload is invalid."):
        super().__init__(
            detail={
                "code": "INVALID_REQUEST",
                "message": message,
            }
        )


class RagCategoryNotFoundError(APIException):
    status_code = status.HTTP_404_NOT_FOUND
    default_code = "category_not_found"

    def __init__(self):
        super().__init__(
            detail={
                "code": "CATEGORY_NOT_FOUND",
                "message": "The category was not found.",
            }
        )


class RagCategoryInactiveError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "category_inactive"

    def __init__(self):
        super().__init__(
            detail={
                "code": "CATEGORY_INACTIVE",
                "message": "The category is inactive.",
            }
        )


class RagServiceUnavailableError(APIException):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_code = "rag_service_unavailable"

    def __init__(self):
        super().__init__(
            detail={
                "code": "RAG_SERVICE_UNAVAILABLE",
                "message": (
                    "The legal question service is unavailable right now."
                ),
            }
        )


class RagInternalError(APIException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_code = "internal_error"

    def __init__(self):
        super().__init__(
            detail={
                "code": "INTERNAL_ERROR",
                "message": (
                    "The legal question could not be processed because of an unexpected error."
                ),
            }
        )
