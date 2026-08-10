from rest_framework import status
from rest_framework.exceptions import APIException


class DocumentCategoryCodeExistsError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "category_code_exists"

    def __init__(self):
        super().__init__(
            detail={
                "code": "CATEGORY_CODE_EXISTS",
                "message": "A document category with this code already exists.",
            }
        )


class DocumentCategoryNameExistsError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "category_name_exists"

    def __init__(self):
        super().__init__(
            detail={
                "code": "CATEGORY_NAME_EXISTS",
                "message": "A document category with this name already exists.",
            }
        )


class DocumentCategoryNotFoundError(APIException):
    status_code = status.HTTP_404_NOT_FOUND
    default_code = "category_not_found"

    def __init__(self):
        super().__init__(
            detail={
                "code": "CATEGORY_NOT_FOUND",
                "message": "The document category was not found.",
            }
        )


class DocumentCategoryInUseError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "category_in_use"

    def __init__(self):
        super().__init__(
            detail={
                "code": "CATEGORY_IN_USE",
                "message": (
                    "This category is referenced by documents and cannot be deleted. "
                    "Deactivate it with is_active=false instead."
                ),
            }
        )
