from __future__ import annotations

from django.core.exceptions import ObjectDoesNotExist
from django.contrib.auth import get_user_model
from rest_framework import serializers

from billing.models import Plan, Subscription
from rag_api.models import QueryHistory


User = get_user_model()

RECENT_QUERY_PREVIEW_LENGTH = 160


class StrictAdminFilterSerializer(serializers.Serializer):
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


class AdminUsersFilterSerializer(StrictAdminFilterSerializer):
    search = serializers.CharField(
        required=False,
        allow_blank=True,
        trim_whitespace=True,
    )
    plan = serializers.CharField(
        required=False,
        allow_blank=True,
        trim_whitespace=True,
    )
    status = serializers.ChoiceField(
        choices=(
            ("active", "Active"),
            ("inactive", "Inactive"),
        ),
        required=False,
    )

    def validate_search(self, value: str) -> str:
        return value.strip()

    def validate_plan(self, value: str) -> str:
        return value.strip()


class AdminDashboardSummarySerializer(serializers.Serializer):
    total_users = serializers.IntegerField()
    paid_subscribers = serializers.IntegerField()
    active_subscribers = serializers.IntegerField()


class AdminDashboardRecentUserSubscriptionSerializer(serializers.Serializer):
    plan_id = serializers.IntegerField(allow_null=True)
    plan_name = serializers.CharField(allow_null=True)
    plan_code = serializers.CharField(allow_null=True)


class AdminDashboardRecentUserSerializer(serializers.ModelSerializer):
    subscription = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "is_active",
            "date_joined",
            "subscription",
        ]

    def get_subscription(self, obj):
        try:
            subscription = obj.subscription
        except ObjectDoesNotExist:
            subscription = None

        if subscription is None:
            return None

        try:
            plan = subscription.plan
        except ObjectDoesNotExist:
            plan = None

        return {
            "plan_id": None if plan is None else plan.id,
            "plan_name": None if plan is None else plan.name,
            "plan_code": None if plan is None else plan.code,
        }


class AdminPlanOverviewSerializer(serializers.Serializer):
    plan_id = serializers.IntegerField(source="id")
    name = serializers.CharField()
    code = serializers.CharField()
    is_active = serializers.BooleanField()
    subscribers_count = serializers.IntegerField()


class AdminDashboardResponseSerializer(serializers.Serializer):
    summary = AdminDashboardSummarySerializer()
    recent_users = AdminDashboardRecentUserSerializer(many=True)
    subscription_overview = AdminPlanOverviewSerializer(many=True)


class AdminUserListSubscriptionSerializer(serializers.Serializer):
    plan_id = serializers.IntegerField(allow_null=True)
    plan_name = serializers.CharField(allow_null=True)
    plan_code = serializers.CharField(allow_null=True)
    status = serializers.CharField(allow_null=True)
    queries_used = serializers.IntegerField(allow_null=True)
    query_limit = serializers.IntegerField(allow_null=True)
    queries_remaining = serializers.IntegerField(allow_null=True)
    usage_reset_at = serializers.DateTimeField(allow_null=True)


class AdminUserListSerializer(serializers.ModelSerializer):
    subscription = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "is_active",
            "date_joined",
            "last_login",
            "subscription",
        ]

    def get_subscription(self, obj):
        try:
            subscription = obj.subscription
        except ObjectDoesNotExist:
            subscription = None

        if subscription is None:
            return None

        try:
            plan = subscription.plan
        except ObjectDoesNotExist:
            plan = None

        return {
            "plan_id": None if plan is None else plan.id,
            "plan_name": None if plan is None else plan.name,
            "plan_code": None if plan is None else plan.code,
            "status": subscription.status,
            "queries_used": subscription.queries_used,
            "query_limit": None if plan is None else plan.query_limit,
            "queries_remaining": (
                0 if plan is None else subscription.queries_remaining
            ),
            "usage_reset_at": subscription.usage_reset_at,
        }


class AdminUserDetailUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "is_active",
            "date_joined",
            "last_login",
        ]


class AdminUserDetailSubscriptionSerializer(serializers.Serializer):
    id = serializers.IntegerField(allow_null=True)
    plan_id = serializers.IntegerField(allow_null=True)
    plan_name = serializers.CharField(allow_null=True)
    plan_code = serializers.CharField(allow_null=True)
    plan_price = serializers.DecimalField(
        max_digits=10,
        decimal_places=2,
        allow_null=True,
    )
    status = serializers.CharField(allow_null=True)
    query_limit = serializers.IntegerField(allow_null=True)
    queries_used = serializers.IntegerField(allow_null=True)
    queries_remaining = serializers.IntegerField(allow_null=True)
    usage_reset_at = serializers.DateTimeField(allow_null=True)
    started_at = serializers.DateTimeField(allow_null=True)
    expires_at = serializers.DateTimeField(allow_null=True)
    cancelled_at = serializers.DateTimeField(allow_null=True)


class AdminUserRecentQuerySerializer(serializers.ModelSerializer):
    answer_preview = serializers.SerializerMethodField()

    class Meta:
        model = QueryHistory
        fields = [
            "id",
            "question",
            "answer_preview",
            "created_at",
        ]

    def get_answer_preview(self, obj):
        answer = (obj.answer or "").strip()

        if len(answer) <= RECENT_QUERY_PREVIEW_LENGTH:
            return answer

        return answer[:RECENT_QUERY_PREVIEW_LENGTH].rstrip() + "..."


class AdminUserRagUsageSerializer(serializers.Serializer):
    total_queries = serializers.IntegerField()
    recent_queries = AdminUserRecentQuerySerializer(many=True)


class AdminUserDetailResponseSerializer(serializers.Serializer):
    user = AdminUserDetailUserSerializer()
    subscription = AdminUserDetailSubscriptionSerializer(allow_null=True)
    rag_usage = AdminUserRagUsageSerializer()
