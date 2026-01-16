"""Auth0-based entitlements (no disk).

This module stores the user's plan in Auth0 user `app_metadata`:
  app_metadata.plan = "free" | "pro"

A FastSpring webhook calls `apply_fastspring_events(...)` to upgrade/downgrade plans.

Required env vars (Render):
  AUTH0_DOMAIN
  AUTH0_MGMT_CLIENT_ID
  AUTH0_MGMT_CLIENT_SECRET

Optional env vars:
  AUTH0_MGMT_AUDIENCE        default: https://<AUTH0_DOMAIN>/api/v2/
  TSA_PRO_SKUS               comma-separated SKUs that grant pro
  FASTSPRING_WEBHOOK_TOKEN   shared secret for webhook (query ?token=... or header X-Webhook-Token)
  PAID_EMAILS                (fallback) comma-separated manual pro users

Notes:
- Auth0 Management API credentials must be Machine-to-Machine (M2M) application.
- Scopes needed:
    read:users, update:users, read:users_by_email
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import os
import time
import json
import urllib.request
import urllib.error
import urllib.parse

from fastapi import HTTPException

# --- Config ---
_AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "").strip()
_AUTH0_MGMT_CLIENT_ID = os.getenv("AUTH0_MGMT_CLIENT_ID", "").strip()
_AUTH0_MGMT_CLIENT_SECRET = os.getenv("AUTH0_MGMT_CLIENT_SECRET", "").strip()
_AUTH0_MGMT_AUDIENCE = os.getenv("AUTH0_MGMT_AUDIENCE", "").strip()
if not _AUTH0_MGMT_AUDIENCE and _AUTH0_DOMAIN:
    _AUTH0_MGMT_AUDIENCE = f"https://{_AUTH0_DOMAIN}/api/v2/"

_FASTSPRING_WEBHOOK_TOKEN = os.getenv("FASTSPRING_WEBHOOK_TOKEN", "").strip()
_TSA_PRO_SKUS = {s.strip() for s in os.getenv("TSA_PRO_SKUS", "").split(",") if s.strip()}

# Optional legacy fallback:
_PAID_EMAILS = {e.strip().lower() for e in os.getenv("PAID_EMAILS", "").split(",") if e.strip()}

# --- Caches ---
_mgmt_token_cache: Dict[str, Any] = {"token": None, "exp": 0}
_plan_cache: Dict[str, Any] = {}  # user_id -> {plan, ts}
_PLAN_CACHE_TTL = 300  # seconds


def mgmt_enabled() -> bool:
    return bool(_AUTH0_DOMAIN and _AUTH0_MGMT_CLIENT_ID and _AUTH0_MGMT_CLIENT_SECRET and _AUTH0_MGMT_AUDIENCE)


def _get_mgmt_token() -> str:
    if not mgmt_enabled():
        raise HTTPException(
            status_code=500,
            detail="Auth0 Management API is not configured. Set AUTH0_MGMT_CLIENT_ID and AUTH0_MGMT_CLIENT_SECRET (and AUTH0_DOMAIN).",
        )

    now = int(time.time())
    tok = _mgmt_token_cache.get("token")
    exp = int(_mgmt_token_cache.get("exp") or 0)
    if tok and now < (exp - 30):
        return str(tok)

    url = f"https://{_AUTH0_DOMAIN}/oauth/token"
    payload = {
        "client_id": _AUTH0_MGMT_CLIENT_ID,
        "client_secret": _AUTH0_MGMT_CLIENT_SECRET,
        "audience": _AUTH0_MGMT_AUDIENCE,
        "grant_type": "client_credentials",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
        token = data.get("access_token")
        expires_in = int(data.get("expires_in") or 3600)
        if not token:
            raise HTTPException(status_code=500, detail=f"Auth0 Management token missing in response: {data}")
        _mgmt_token_cache["token"] = token
        _mgmt_token_cache["exp"] = now + expires_in
        return str(token)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=500, detail=f"Auth0 Management token error {e.code}: {raw[:1200]}")


def _mgmt_get(path: str) -> Any:
    token = _get_mgmt_token()
    url = f"https://{_AUTH0_DOMAIN}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _mgmt_patch(path: str, body: Dict[str, Any]) -> Any:
    token = _get_mgmt_token()
    url = f"https://{_AUTH0_DOMAIN}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="PATCH",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def auth0_user_id_by_email(email: str) -> Optional[str]:
    if not email:
        return None
    q = urllib.parse.urlencode({"email": email})
    data = _mgmt_get(f"/api/v2/users-by-email?{q}")
    if isinstance(data, list) and data:
        return data[0].get("user_id")
    return None


def get_plan_from_auth0(user_id: str) -> Optional[str]:
    if not user_id or not mgmt_enabled():
        return None

    now = time.time()
    cached = _plan_cache.get(user_id)
    if cached and (now - float(cached.get("ts") or 0)) < _PLAN_CACHE_TTL:
        plan = cached.get("plan")
        return str(plan).strip().lower() if plan else None

    try:
        u = _mgmt_get(
            f"/api/v2/users/{urllib.parse.quote(user_id, safe='')}?fields=app_metadata&include_fields=true"
        )
        plan = (u.get("app_metadata") or {}).get("plan")
        plan = str(plan).strip().lower() if plan else None
        _plan_cache[user_id] = {"plan": plan, "ts": now}
        return plan
    except Exception:
        return None


def set_plan_in_auth0(user_id: str, plan: str, meta: Optional[Dict[str, Any]] = None) -> None:
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing Auth0 user_id")
    plan = (plan or "free").strip().lower()
    meta = meta or {}
    app_md = {"plan": plan, **meta, "updated_at": int(time.time())}
    _mgmt_patch(f"/api/v2/users/{urllib.parse.quote(user_id, safe='')}", {"app_metadata": app_md})
    _plan_cache[user_id] = {"plan": plan, "ts": time.time()}


def get_user_plan(email: Optional[str], user_id: Optional[str]) -> str:
    # Prefer Auth0 metadata
    if user_id:
        p = get_plan_from_auth0(user_id)
        if p in ("free", "pro"):
            return p

    # Legacy fallback
    if email and email.strip().lower() in _PAID_EMAILS:
        return "pro"

    return "free"


def require_pro_user(user_claims: Dict[str, Any]) -> None:
    user_id = user_claims.get("sub")
    email = user_claims.get("email")
    plan = get_user_plan(email=email, user_id=user_id)
    if plan != "pro":
        raise HTTPException(status_code=403, detail="Upgrade required: this feature is available on the Pro plan.")


def validate_fastspring_token(token_q: Optional[str], token_h: Optional[str]) -> None:
    if not _FASTSPRING_WEBHOOK_TOKEN:
        return
    if token_q != _FASTSPRING_WEBHOOK_TOKEN and token_h != _FASTSPRING_WEBHOOK_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized webhook")


def _decide_plan_from_event(etype: str, data: Dict[str, Any]) -> Optional[str]:
    etype = (etype or "").strip().lower()

    # Subscription lifecycle events
    if etype in ("subscription.activated", "subscription.started", "subscription.charge.completed", "subscription.updated"):
        return "pro"

    if etype in ("subscription.canceled", "subscription.deactivated"):
        return "free"

    if etype == "order.completed":
        items = data.get("items") or []
        try:
            # If SKU list is configured, use it
            if _TSA_PRO_SKUS:
                skus = {str((it.get("sku") or it.get("product") or "")).strip() for it in items}
                if skus & _TSA_PRO_SKUS:
                    return "pro"
            # Otherwise, treat any subscription product as pro
            if any(bool(it.get("isSubscription")) for it in items):
                return "pro"
        except Exception:
            return None

    return None


def apply_fastspring_events(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply webhook events to Auth0 app_metadata. Returns a list of updates."""
    if not mgmt_enabled():
        raise HTTPException(status_code=500, detail="Auth0 Management API not configured.")

    events = payload.get("events") or []
    updates: List[Dict[str, Any]] = []

    for ev in events:
        etype = ev.get("type") or ""
        data = ev.get("data") or {}
        customer = data.get("customer") or {}
        email = customer.get("email")
        if not email:
            # Some events might carry it elsewhere; try recipients
            recips = data.get("recipients") or []
            for r in recips:
                rec = (r or {}).get("recipient") or {}
                email = rec.get("email") or email
                if email:
                    break

        if not email:
            continue

        plan = _decide_plan_from_event(etype, data)
        if not plan:
            continue

        user_id = auth0_user_id_by_email(str(email))
        if not user_id:
            # User might not have logged in yet; nothing we can do without persistence.
            updates.append({"email": email, "plan": plan, "ok": False, "reason": "auth0_user_not_found"})
            continue

        order_id = data.get("id") or data.get("order")
        sub_id = None
        # Try to extract subscription id from items
        for it in (data.get("items") or []):
            if it.get("subscription"):
                sub_id = it.get("subscription")
                break

        meta = {
            "source": f"fastspring:{etype}",
            "fastspring_order_id": order_id,
            "fastspring_subscription_id": sub_id,
        }

        set_plan_in_auth0(user_id, plan, meta=meta)
        updates.append({"email": email, "user_id": user_id, "plan": plan, "ok": True})

    return updates
