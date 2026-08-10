import logging

from django.db import DatabaseError, IntegrityError, transaction
from django.db.models import Q

from billing.exceptions import PlanCodeExistsError, PlanNotFoundError
from billing.models import Plan, Subscription


logger = logging.getLogger(__name__)


def create_plan(*, validated_data: dict) -> Plan:
    code = validated_data["code"].strip().upper()
    payload = dict(validated_data)
    payload["code"] = code

    try:
        with transaction.atomic():
            if Plan.objects.filter(code__iexact=code).exists():
                raise PlanCodeExistsError()

            plan = Plan.objects.create(**payload)
    except PlanCodeExistsError:
        raise
    except IntegrityError as exception:
        logger.warning(
            "Integrity error while creating billing plan code=%s",
            code,
        )
        raise PlanCodeExistsError() from exception
    except DatabaseError:
        logger.exception("Database error while creating billing plan code=%s", code)
        raise

    return plan


def update_plan(*, plan_id: int, validated_data: dict) -> Plan:
    try:
        with transaction.atomic():
            try:
                plan = Plan.objects.select_for_update().get(pk=plan_id)
            except Plan.DoesNotExist as exception:
                raise PlanNotFoundError() from exception

            for field, value in validated_data.items():
                setattr(plan, field, value)

            try:
                plan.save()
            except IntegrityError as exception:
                logger.warning(
                    "Integrity error while updating billing plan id=%s",
                    plan_id,
                )
                raise PlanCodeExistsError() from exception
    except PlanNotFoundError:
        raise
    except DatabaseError:
        logger.exception("Database error while updating billing plan id=%s", plan_id)
        raise

    return plan


def get_admin_subscriptions(*, filters: dict):
    queryset = Subscription.objects.select_related("user", "plan").order_by(
        "-created_at",
        "-id",
    )

    status = filters.get("status")
    if status:
        queryset = queryset.filter(status=status)

    plan_code = filters.get("plan")
    if plan_code:
        queryset = queryset.filter(plan__code__iexact=plan_code.strip())

    search = filters.get("search")
    if search:
        queryset = queryset.filter(
            Q(user__username__icontains=search)
            | Q(user__email__icontains=search)
        )

    return queryset
