from django.urls import path

from billing.controllers.billing_controller import (
    PlansController,
    SubscriptionController,
)


app_name = "billing"


urlpatterns = [
    path(
        "plans/",
        PlansController.as_view(),
        name="plans",
    ),
    path(
        "subscription/",
        SubscriptionController.as_view(),
        name="subscription",
    ),
]
