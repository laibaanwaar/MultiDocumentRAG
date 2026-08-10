from django.urls import path

from documents.controllers.category_controller import DocumentCategoryController
from documents.controllers.legal_document_controller import LegalDocumentController


app_name = "documents"


urlpatterns = [
    path(
        "documents/",
        LegalDocumentController.as_view(),
        name="documents",
    ),
    path(
        "documents/<int:document_id>/",
        LegalDocumentController.as_view(),
        name="document-detail",
    ),
    path(
        "document-categories/",
        DocumentCategoryController.as_view(),
        name="document-categories",
    ),
    path(
        "document-categories/<int:category_id>/",
        DocumentCategoryController.as_view(),
        name="document-category-detail",
    ),
]
