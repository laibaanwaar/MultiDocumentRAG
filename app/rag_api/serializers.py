from django.conf import settings
from rest_framework import serializers

from documents.models import DocumentCategory
from rag_api.models import QueryHistory


class StrictQueryInputSerializer(serializers.Serializer):
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


class RagQueryRequestSerializer(StrictQueryInputSerializer):
    question = serializers.CharField(
        required=True,
        allow_blank=False,
        trim_whitespace=True,
        max_length=getattr(settings, "RAG_QUERY_MAX_LENGTH", 4000),
    )
    category_id = serializers.IntegerField(
        required=False,
        min_value=1,
    )

    def validate_question(self, value):
        raw_value = self.initial_data.get("question")

        if not isinstance(raw_value, str):
            raise serializers.ValidationError("This field must be a string.")

        normalized = value.strip()

        if not normalized:
            raise serializers.ValidationError("This field may not be blank.")

        if len(normalized) > getattr(settings, "RAG_QUERY_MAX_LENGTH", 4000):
            raise serializers.ValidationError(
                "This field may not be longer than the configured limit."
            )

        return normalized


class QueryHistoryCategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentCategory
        fields = [
            "id",
            "name",
            "code",
        ]


class QueryHistorySerializer(serializers.ModelSerializer):
    category = QueryHistoryCategorySerializer(read_only=True)

    class Meta:
        model = QueryHistory
        fields = [
            "id",
            "question",
            "answer",
            "category",
            "sources",
            "created_at",
        ]
