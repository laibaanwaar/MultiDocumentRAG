from django.urls import path

from rag_api.controllers.history_controller import RagHistoryController
from rag_api.controllers.query_controller import RagQueryController


app_name = "rag_api"


urlpatterns = [
    path(
        "query/",
        RagQueryController.as_view(),
        name="query",
    ),
    path(
        "history/",
        RagHistoryController.as_view(),
        name="history",
    ),
]
