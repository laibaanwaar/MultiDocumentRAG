from django.urls import path

from accounts.controllers.auth_controller import (
    LoginController,
    MeController,
    LogoutController,
    RefreshTokenController,
    SignupController,
    ResendOTPController,
    VerifyEmailController,
)


app_name = "accounts"


urlpatterns = [
    path(
        "login/",
        LoginController.as_view(),
        name="login",
    ),
    path(
        "refresh/",
        RefreshTokenController.as_view(),
        name="refresh",
    ),
    path(
        "logout/",
        LogoutController.as_view(),
        name="logout",
    ),
    path(
        "me/",
        MeController.as_view(),
        name="me",
    ),
    path(
        "signup/",
        SignupController.as_view(),
        name="signup",
    ),
    path(
        "resend-otp/",
        ResendOTPController.as_view(),
        name="resend-otp",
    ),
    path(
        "verify-email/",
        VerifyEmailController.as_view(),
        name="verify-email",
    ),
]
