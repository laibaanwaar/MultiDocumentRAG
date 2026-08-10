from rest_framework import status
from rest_framework.exceptions import APIException


class BillingConfigurationError(APIException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_code = "billing_configuration_error"

    def __init__(self):
        super().__init__(
            detail={
                "code": "BILLING_CONFIGURATION_ERROR",
                "message": (
                    "The billing configuration is unavailable right now."
                ),
            }
        )


class SubscriptionNotFoundError(APIException):
    status_code = status.HTTP_404_NOT_FOUND
    default_code = "subscription_not_found"

    def __init__(self):
        super().__init__(
            detail={
                "code": "SUBSCRIPTION_NOT_FOUND",
                "message": "No subscription was found for this account.",
            }
        )


class SubscriptionInactiveError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "subscription_inactive"

    def __init__(self):
        super().__init__(
            detail={
                "code": "SUBSCRIPTION_INACTIVE",
                "message": "The subscription is inactive.",
            }
        )


class SubscriptionExpiredError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "subscription_expired"

    def __init__(self):
        super().__init__(
            detail={
                "code": "SUBSCRIPTION_EXPIRED",
                "message": "The subscription has expired.",
            }
        )


class PlanInactiveError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "plan_inactive"

    def __init__(self):
        super().__init__(
            detail={
                "code": "PLAN_INACTIVE",
                "message": "The assigned plan is inactive.",
            }
        )


class DuplicateSubscriptionError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "duplicate_subscription"

    def __init__(self):
        super().__init__(
            detail={
                "code": "SUBSCRIPTION_CONFLICT",
                "message": (
                    "More than one subscription exists for this account."
                ),
            }
        )


class PlanCodeExistsError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "plan_code_exists"

    def __init__(self):
        super().__init__(
            detail={
                "code": "PLAN_CODE_EXISTS",
                "message": "A plan with this code already exists.",
            }
        )


class PlanNotFoundError(APIException):
    status_code = status.HTTP_404_NOT_FOUND
    default_code = "plan_not_found"

    def __init__(self):
        super().__init__(
            detail={
                "code": "PLAN_NOT_FOUND",
                "message": "The plan was not found.",
            }
        )


class EmptyUpdateError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "empty_update"

    def __init__(self):
        super().__init__(
            detail={
                "code": "EMPTY_UPDATE",
                "message": "The update request must include at least one field.",
            }
        )


class ImmutableFieldError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "immutable_field"

    def __init__(self):
        super().__init__(
            detail={
                "code": "IMMUTABLE_FIELD",
                "message": "The code field cannot be updated.",
            }
        )
