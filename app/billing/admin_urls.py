from django.urls import path

from billing.controllers.admin_billing_controller import (
    AdminBillingPlansController,
    AdminSubscriptionsController,
)


app_name = "billing_admin"


urlpatterns = [
    path(
        "billing/plans/",
        AdminBillingPlansController.as_view(),
        name="admin-plans",
    ),
    path(
        "billing/plans/<int:plan_id>/",
        AdminBillingPlansController.as_view(),
        name="admin-plan-detail",
    ),
    path(
        "subscriptions/",
        AdminSubscriptionsController.as_view(),
        name="admin-subscriptions",
    ),
]
