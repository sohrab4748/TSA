import os
from fastapi import APIRouter, Request, HTTPException
from app.entitlements_db import init_db, set_plan

router = APIRouter()

# Call once at import time
init_db()

PRO_PRODUCT_SKUS = set(filter(None, os.getenv("TSA_PRO_SKUS", "").split(",")))
# Example: TSA_PRO_MONTHLY,TSA_PRO_YEARLY

@router.post("/webhooks/fastspring")
async def fastspring_webhook(request: Request):
    # Simple protection: require a shared secret token in URL query
    token = request.query_params.get("token")
    if not token or token != os.getenv("FASTSPRING_WEBHOOK_TOKEN"):
        raise HTTPException(status_code=401, detail="Unauthorized webhook")

    payload = await request.json()
    events = payload.get("events", [])

    for ev in events:
        etype = ev.get("type")
        data = ev.get("data", {}) or {}
        customer = data.get("customer", {}) or {}
        email = customer.get("email")

        if not email:
            continue

        order_id = data.get("id") or data.get("order")
        plan = None
        sub_id = None

        # 1) If it's an order, decide plan based on purchased SKU(s)
        if etype == "order.completed":
            items = data.get("items", []) or []
            skus = { (it.get("sku") or it.get("product") or "").strip() for it in items }
            # if any sku matches PRO list => pro
            if skus & PRO_PRODUCT_SKUS:
                plan = "pro"
            else:
                # If you sell only one subscription product for TSA, you can also treat any subscription item as pro:
                if any(it.get("isSubscription") for it in items):
                    plan = "pro"
            # capture subscription id if present
            for it in items:
                if it.get("subscription"):
                    sub_id = it.get("subscription")
                    break

        # 2) Subscription lifecycle events can upgrade/downgrade
        if etype in ("subscription.activated", "subscription.charge.completed", "subscription.updated"):
            plan = plan or "pro"
            sub_id = sub_id or data.get("id")

        if etype in ("subscription.canceled", "subscription.deactivated"):
            plan = "free"
            sub_id = sub_id or data.get("id")

        if plan:
            set_plan(email=email, plan=plan, source=f"fastspring:{etype}", order_id=order_id, subscription_id=sub_id)

    return {"ok": True}
