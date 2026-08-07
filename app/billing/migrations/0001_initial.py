# Generated manually for the billing app.

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Plan",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=120)),
                ("code", models.CharField(max_length=50, unique=True)),
                (
                    "price",
                    models.DecimalField(decimal_places=2, max_digits=10),
                ),
                (
                    "billing_period",
                    models.CharField(
                        choices=[
                            ("monthly", "Monthly"),
                            ("yearly", "Yearly"),
                        ],
                        max_length=20,
                    ),
                ),
                ("document_limit", models.PositiveIntegerField()),
                ("query_limit", models.PositiveIntegerField()),
                ("is_active", models.BooleanField(default=True)),
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True),
                ),
                (
                    "updated_at",
                    models.DateTimeField(auto_now=True),
                ),
            ],
            options={
                "ordering": ["id"],
            },
        ),
        migrations.CreateModel(
            name="Subscription",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("active", "Active"),
                            ("canceled", "Canceled"),
                            ("expired", "Expired"),
                        ],
                        default="active",
                        max_length=20,
                    ),
                ),
                ("queries_used", models.PositiveIntegerField(default=0)),
                (
                    "usage_reset_at",
                    models.DateTimeField(blank=True, null=True),
                ),
                (
                    "started_at",
                    models.DateTimeField(default=django.utils.timezone.now),
                ),
                (
                    "expires_at",
                    models.DateTimeField(blank=True, null=True),
                ),
                (
                    "cancelled_at",
                    models.DateTimeField(blank=True, null=True),
                ),
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True),
                ),
                (
                    "updated_at",
                    models.DateTimeField(auto_now=True),
                ),
                (
                    "plan",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="subscriptions",
                        to="billing.plan",
                    ),
                ),
                (
                    "user",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="subscription",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
    ]
