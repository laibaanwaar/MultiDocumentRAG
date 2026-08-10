from django.urls import path

from accounts.controllers.admin_controller import (
    AdminDashboardController,
    AdminUserDetailController,
    AdminUsersController,
)


app_name = "accounts_admin"


urlpatterns = [
    path(
        "dashboard/",
        AdminDashboardController.as_view(),
        name="admin-dashboard",
    ),
    path(
        "users/",
        AdminUsersController.as_view(),
        name="admin-users",
    ),
    path(
        "users/<int:user_id>/",
        AdminUserDetailController.as_view(),
        name="admin-user-detail",
    ),
]
