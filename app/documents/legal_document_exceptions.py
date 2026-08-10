from rest_framework import status
from rest_framework.exceptions import APIException


class LegalDocumentNotFoundError(APIException):
    status_code = status.HTTP_404_NOT_FOUND
    default_code = "document_not_found"

    def __init__(self):
        super().__init__(
            detail={
                "code": "DOCUMENT_NOT_FOUND",
                "message": "The legal document was not found.",
            }
        )


class LegalDocumentAlreadyExistsError(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_code = "document_already_exists"

    def __init__(self):
        super().__init__(
            detail={
                "code": "DOCUMENT_ALREADY_EXISTS",
                "message": "A legal document with this checksum already exists.",
            }
        )


class LegalDocumentCategoryInactiveError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "category_inactive"

    def __init__(self):
        super().__init__(
            detail={
                "code": "CATEGORY_INACTIVE",
                "message": "The selected category is inactive.",
            }
        )


class LegalDocumentInvalidFileTypeError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "invalid_file_type"

    def __init__(self):
        super().__init__(
            detail={
                "code": "INVALID_FILE_TYPE",
                "message": "Only PDF files are allowed.",
            }
        )


class LegalDocumentInvalidPdfError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "invalid_pdf"

    def __init__(self):
        super().__init__(
            detail={
                "code": "INVALID_PDF",
                "message": "The uploaded file is not a valid PDF document.",
            }
        )


class LegalDocumentFileTooLargeError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "file_too_large"

    def __init__(self):
        super().__init__(
            detail={
                "code": "FILE_TOO_LARGE",
                "message": "The uploaded file exceeds the maximum allowed size.",
            }
        )


class LegalDocumentImmutableFieldError(APIException):
    status_code = status.HTTP_400_BAD_REQUEST
    default_code = "immutable_field"

    def __init__(self):
        super().__init__(
            detail={
                "code": "IMMUTABLE_FIELD",
                "message": "One or more immutable fields were provided.",
            }
        )


class LegalDocumentStorageError(APIException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_code = "storage_error"

    def __init__(self):
        super().__init__(
            detail={
                "code": "STORAGE_ERROR",
                "message": "The legal document could not be stored right now.",
            }
        )
