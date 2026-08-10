from rest_framework.permissions import BasePermission


class IsStaffUser(BasePermission):
    message = "Admin access is required."

    def has_permission(self, request, view):
        return bool(getattr(request, "user", None) and request.user.is_staff)
