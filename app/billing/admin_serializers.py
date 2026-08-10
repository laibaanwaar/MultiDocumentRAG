from decimal import Decimal

from django.contrib.auth import get_user_model
from rest_framework import serializers

from billing.models import Plan
from billing.models import Subscription


User = get_user_model()


class StrictPlanInputSerializer(serializers.Serializer):
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


class AdminPlanCreateSerializer(StrictPlanInputSerializer):
    name = serializers.CharField(max_length=120, allow_blank=False, trim_whitespace=True)
    code = serializers.CharField(max_length=50, allow_blank=False, trim_whitespace=True)
    price = serializers.DecimalField(
        max_digits=10,
        decimal_places=2,
        min_value=Decimal("0"),
    )
    billing_period = serializers.ChoiceField(choices=Plan.BillingPeriod.choices)
    document_limit = serializers.IntegerField(min_value=0)
    query_limit = serializers.IntegerField(min_value=0)
    is_active = serializers.BooleanField()

    def validate_code(self, value):
        normalized = value.strip().upper()
        if not normalized:
            raise serializers.ValidationError("This field may not be blank.")
        return normalized


class AdminPlanUpdateSerializer(StrictPlanInputSerializer):
    name = serializers.CharField(
        max_length=120,
        allow_blank=False,
        required=False,
        trim_whitespace=True,
    )
    price = serializers.DecimalField(
        max_digits=10,
        decimal_places=2,
        min_value=Decimal("0"),
        required=False,
    )
    billing_period = serializers.ChoiceField(
        choices=Plan.BillingPeriod.choices,
        required=False,
    )
    document_limit = serializers.IntegerField(min_value=0, required=False)
    query_limit = serializers.IntegerField(min_value=0, required=False)
    is_active = serializers.BooleanField(required=False)


class AdminPlanResponseSerializer(serializers.ModelSerializer):
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
        ]


class AdminSubscriptionFilterSerializer(serializers.Serializer):
    status = serializers.ChoiceField(
        choices=Subscription.Status.choices,
        required=False,
    )
    plan = serializers.CharField(
        required=False,
        allow_blank=True,
        trim_whitespace=True,
    )
    search = serializers.CharField(
        required=False,
        allow_blank=True,
        trim_whitespace=True,
    )


class AdminSubscriptionUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "is_active",
        ]


class AdminSubscriptionPlanSerializer(serializers.ModelSerializer):
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
        ]


class AdminSubscriptionListSerializer(serializers.ModelSerializer):
    user = AdminSubscriptionUserSerializer()
    plan = AdminSubscriptionPlanSerializer()
    queries_remaining = serializers.SerializerMethodField()

    class Meta:
        model = Subscription
        fields = [
            "id",
            "user",
            "plan",
            "status",
            "queries_used",
            "queries_remaining",
            "usage_reset_at",
            "started_at",
            "expires_at",
            "cancelled_at",
        ]

    def get_queries_remaining(self, obj):
        return obj.queries_remaining
