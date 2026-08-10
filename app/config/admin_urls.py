from django.urls import include, path


urlpatterns = [
    path("", include("accounts.admin_urls")),
    path("", include("billing.admin_urls")),
]
