from django.core.mail import send_mail


def send_signup_otp(
    *,
    recipient_email: str,
    otp_code: str,
) -> None:
    subject = "Verify your Legal RAG account"

    message = (
        "Welcome to Legal RAG.\n\n"
        f"Your email verification code is: {otp_code}\n\n"
        "This code will expire in 30 minutes.\n"
        "Do not share this code with anyone.\n\n"
        "If you did not create this account, "
        "you can ignore this email."
    )

    delivered_count = send_mail(
        subject=subject,
        message=message,
        from_email=None,
        recipient_list=[recipient_email],
        fail_silently=False,
    )

    if delivered_count != 1:
        raise RuntimeError(
            "Email provider did not confirm delivery."
        )
