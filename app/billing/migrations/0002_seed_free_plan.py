from django.db import migrations


def seed_free_plan(apps, schema_editor):
    Plan = apps.get_model("billing", "Plan")

    Plan.objects.get_or_create(
        code="FREE",
        defaults={
            "name": "Free",
            "price": "0.00",
            "billing_period": "monthly",
            "document_limit": 3,
            "query_limit": 30,
            "is_active": True,
        },
    )


def unseed_free_plan(apps, schema_editor):
    Plan = apps.get_model("billing", "Plan")
    Plan.objects.filter(code="FREE").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("billing", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(seed_free_plan, unseed_free_plan),
    ]
