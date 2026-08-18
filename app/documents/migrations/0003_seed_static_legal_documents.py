from __future__ import annotations

import hashlib
from pathlib import Path

from django.conf import settings
from django.db import migrations


STATIC_CATEGORIES = (
    {
        "code": "PPC",
        "name": "Pakistan Penal Code",
        "description": "Pakistan Penal Code, 1860.",
    },
    {
        "code": "CONST",
        "name": "Constitution of Pakistan",
        "description": "Constitution of Pakistan, 1973.",
    },
    {
        "code": "ATA",
        "name": "Anti-Terrorism Act",
        "description": "Anti-Terrorism Act, 1997.",
    },
    {
        "code": "AMLA",
        "name": "Anti-Money Laundering Act",
        "description": "Anti-Money Laundering Act, 2010.",
    },
    {
        "code": "OTHER",
        "name": "Regulatory / Other Law",
        "description": "Catch-all category for regulatory or other legal documents.",
    },
)


STATIC_DOCUMENTS = (
    {
        "title": "Pakistan Penal Code, 1860",
        "category_code": "PPC",
        "file_name": "Pakistan Penal Code.pdf",
        "original_filename": "Pakistan Penal Code.pdf",
        "checksum_label": "ppc_1860",
    },
    {
        "title": "Constitution of the Islamic Republic of Pakistan, 1973",
        "category_code": "CONST",
        "file_name": "Constitution_of_pakistan.pdf",
        "original_filename": "Constitution_of_pakistan.pdf",
        "checksum_label": "constitution_1973",
    },
    {
        "title": "Anti-Terrorism Act, 1997",
        "category_code": "ATA",
        "file_name": "THE ANTI-TERRORISM ACT, 1997.pdf",
        "original_filename": "THE ANTI-TERRORISM ACT, 1997.pdf",
        "checksum_label": "ata_1997",
    },
    {
        "title": "Anti-Money Laundering Act, 2010",
        "category_code": "AMLA",
        "file_name": "ANTI-MONEY LAUNDERING ACT, 2010.pdf",
        "original_filename": "ANTI-MONEY LAUNDERING ACT, 2010.pdf",
        "checksum_label": "amla_2010",
    },
)


def _seed_user_id(apps) -> int | None:
    User = apps.get_model(settings.AUTH_USER_MODEL)

    staff_user = (
        User.objects.filter(is_staff=True)
        .order_by("id")
        .first()
    )
    if staff_user is not None:
        return staff_user.id

    superuser = (
        User.objects.filter(is_superuser=True)
        .order_by("id")
        .first()
    )
    if superuser is not None:
        return superuser.id

    any_user = User.objects.order_by("id").first()
    if any_user is not None:
        return any_user.id

    return None


def _checksum_for_file(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def seed_static_documents(apps, schema_editor):
    DocumentCategory = apps.get_model("documents", "DocumentCategory")
    LegalDocument = apps.get_model("documents", "LegalDocument")

    media_root = Path(settings.MEDIA_ROOT)
    document_root = media_root

    categories_by_code = {}

    for category_seed in STATIC_CATEGORIES:
        category, _created = DocumentCategory.objects.update_or_create(
            code=category_seed["code"],
            defaults={
                "name": category_seed["name"],
                "description": category_seed["description"],
                "is_active": True,
            },
        )
        categories_by_code[category.code] = category

    uploaded_by_id = _seed_user_id(apps)
    if uploaded_by_id is None:
        return

    for document_seed in STATIC_DOCUMENTS:
        file_path = document_root / document_seed["file_name"]

        if not file_path.exists():
            raise RuntimeError(
                "Static legal document PDF is missing: "
                f"{file_path}"
            )

        checksum = _checksum_for_file(file_path)
        category = categories_by_code[document_seed["category_code"]]
        relative_file_name = document_seed["file_name"]

        LegalDocument.objects.update_or_create(
            checksum_sha256=checksum,
            defaults={
                "title": document_seed["title"],
                "category": category,
                "file": relative_file_name,
                "original_filename": document_seed["original_filename"],
                "file_size": file_path.stat().st_size,
                "content_type": "application/pdf",
                "status": "READY",
                "uploaded_by_id": uploaded_by_id,
                "ingestion_error": None,
            },
        )


def unseed_static_documents(apps, schema_editor):
    DocumentCategory = apps.get_model("documents", "DocumentCategory")
    LegalDocument = apps.get_model("documents", "LegalDocument")

    for document_seed in STATIC_DOCUMENTS:
        LegalDocument.objects.filter(
            title=document_seed["title"],
            original_filename=document_seed["original_filename"],
            status="READY",
        ).delete()

    DocumentCategory.objects.filter(
        code__in=[seed["code"] for seed in STATIC_CATEGORIES]
    ).delete()


class Migration(migrations.Migration):
    dependencies = [
        ("documents", "0002_legaldocument"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RunPython(
            seed_static_documents,
            reverse_code=unseed_static_documents,
        ),
    ]
