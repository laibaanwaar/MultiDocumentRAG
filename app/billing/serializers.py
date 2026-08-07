from rest_framework import serializers

from billing.models import Plan, Subscription


class PlanSerializer(serializers.ModelSerializer):
    class Meta:
        model = Plan
        fields = [
            "id",
            "name",
            "code",
            "price",
            "billing_period",
            "document_limit",
            "query_limit",
            "is_active",
            "created_at",
            "updated_at",
        ]


class SubscriptionSerializer(serializers.ModelSerializer):
    plan = PlanSerializer()
    queries_remaining = serializers.IntegerField(read_only=True)

    class Meta:
        model = Subscription
        fields = [
            "id",
            "plan",
            "status",
            "queries_used",
            "queries_remaining",
            "usage_reset_at",
            "started_at",
            "expires_at",
            "cancelled_at",
            "created_at",
            "updated_at",
        ]
