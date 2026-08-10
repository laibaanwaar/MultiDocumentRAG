from rest_framework import serializers
from django.contrib.auth import get_user_model

from documents.models import DocumentCategory, LegalDocument


User = get_user_model()


class StrictLegalDocumentInputSerializer(serializers.Serializer):
    def validate(self, attrs):
        allowed_fields = set(self.fields.keys())
        incoming_fields = set(self.initial_data.keys())
        unknown_fields = sorted(incoming_fields - allowed_fields)

        if unknown_fields:
            raise serializers.ValidationError(
                {
                    field: ["This field is not allowed."]
                    for field in unknown_fields
                }
            )

        return attrs


class LegalDocumentCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentCategory
        fields = [
            "id",
            "name",
            "code",
            "description",
            "is_active",
        ]


class LegalDocumentUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
        ]


class LegalDocumentCreateSerializer(StrictLegalDocumentInputSerializer):
    title = serializers.CharField(
        max_length=255,
        allow_blank=False,
        trim_whitespace=True,
    )
    category_id = serializers.IntegerField(min_value=1)
    file = serializers.FileField()

    def validate_title(self, value):
        normalized = value.strip()

        if not normalized:
            raise serializers.ValidationError("This field may not be blank.")

        return normalized


class LegalDocumentUpdateSerializer(StrictLegalDocumentInputSerializer):
    title = serializers.CharField(
        max_length=255,
        allow_blank=False,
        required=False,
        trim_whitespace=True,
    )
    category_id = serializers.IntegerField(
        min_value=1,
        required=False,
    )

    def validate_title(self, value):
        normalized = value.strip()

        if not normalized:
            raise serializers.ValidationError("This field may not be blank.")

        return normalized


class LegalDocumentFilterSerializer(StrictLegalDocumentInputSerializer):
    page = serializers.IntegerField(
        required=False,
        min_value=1,
    )
    category = serializers.CharField(
        required=False,
        allow_blank=False,
        trim_whitespace=True,
    )
    status = serializers.CharField(
        required=False,
        allow_blank=False,
        trim_whitespace=True,
    )
    search = serializers.CharField(
        required=False,
        allow_blank=False,
        trim_whitespace=True,
    )

    def validate_status(self, value):
        normalized = value.strip().upper()
        allowed_statuses = {choice for choice, _label in LegalDocument.Status.choices}

        if normalized not in allowed_statuses:
            raise serializers.ValidationError("Invalid status.")

        return normalized


class LegalDocumentListSerializer(serializers.ModelSerializer):
    category = LegalDocumentCategorySerializer(read_only=True)
    uploaded_by = LegalDocumentUserSerializer(read_only=True)

    class Meta:
        model = LegalDocument
        fields = [
            "id",
            "title",
            "category",
            "original_filename",
            "file_size",
            "content_type",
            "status",
            "uploaded_by",
            "created_at",
            "updated_at",
        ]


class LegalDocumentDetailSerializer(serializers.ModelSerializer):
    category = LegalDocumentCategorySerializer(read_only=True)
    uploaded_by = LegalDocumentUserSerializer(read_only=True)

    class Meta:
        model = LegalDocument
        fields = [
            "id",
            "title",
            "category",
            "original_filename",
            "file",
            "file_size",
            "content_type",
            "checksum_sha256",
            "status",
            "uploaded_by",
            "ingestion_error",
            "created_at",
            "updated_at",
        ]


class LegalDocumentResponseSerializer(LegalDocumentDetailSerializer):
    class Meta(LegalDocumentDetailSerializer.Meta):
        fields = LegalDocumentDetailSerializer.Meta.fields
