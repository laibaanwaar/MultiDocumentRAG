from rest_framework import serializers

from documents.models import DocumentCategory


class StrictDocumentCategoryInputSerializer(serializers.Serializer):
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


class DocumentCategoryCreateSerializer(StrictDocumentCategoryInputSerializer):
    name = serializers.CharField(max_length=120, allow_blank=False, trim_whitespace=True)
    code = serializers.CharField(max_length=50, allow_blank=False, trim_whitespace=True)
    description = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=False,
    )
    is_active = serializers.BooleanField(required=False, default=True)

    def validate_name(self, value):
        normalized = value.strip()
        if not normalized:
            raise serializers.ValidationError("This field may not be blank.")
        return normalized

    def validate_code(self, value):
        normalized = value.strip().upper()
        if not normalized:
            raise serializers.ValidationError("This field may not be blank.")
        return normalized

    def validate_description(self, value):
        return value or ""


class DocumentCategoryUpdateSerializer(StrictDocumentCategoryInputSerializer):
    name = serializers.CharField(
        max_length=120,
        allow_blank=False,
        required=False,
        trim_whitespace=True,
    )
    description = serializers.CharField(
        required=False,
        allow_blank=True,
        allow_null=False,
    )
    is_active = serializers.BooleanField(required=False)

    def validate_name(self, value):
        normalized = value.strip()
        if not normalized:
            raise serializers.ValidationError("This field may not be blank.")
        return normalized

    def validate_description(self, value):
        return value or ""


class DocumentCategoryListSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentCategory
        fields = [
            "id",
            "name",
            "code",
            "description",
            "is_active",
        ]


class DocumentCategoryResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentCategory
        fields = [
            "id",
            "name",
            "code",
            "description",
            "is_active",
            "created_at",
            "updated_at",
        ]
