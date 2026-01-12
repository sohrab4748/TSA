import os, json, time
import urllib.request
from typing import Dict, Any, Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "").strip()
AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", "").strip()
AUTH0_ISSUER = os.getenv("AUTH0_ISSUER", "").strip()
ALGORITHMS = ["RS256"]

_bearer = HTTPBearer(auto_error=False)

_jwks_cache: Dict[str, Any] = {"ts": 0.0, "jwks": None}
_JWKS_TTL_SECONDS = 3600


def _get_jwks() -> Dict[str, Any]:
    if not AUTH0_DOMAIN:
        raise RuntimeError("Missing AUTH0_DOMAIN env var")

    now = time.time()
    if _jwks_cache["jwks"] is not None and (now - _jwks_cache["ts"]) < _JWKS_TTL_SECONDS:
        return _jwks_cache["jwks"]

    url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))

    _jwks_cache["jwks"] = data
    _jwks_cache["ts"] = now
    return data


def _get_rsa_key(token: str) -> Optional[Dict[str, Any]]:
    jwks = _get_jwks()
    headers = jwt.get_unverified_header(token)
    kid = headers.get("kid")
    if not kid:
        return None

    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key
    return None


def require_auth(payload: HTTPAuthorizationCredentials = Security(_bearer)) -> Dict[str, Any]:
    if payload is None or (payload.scheme or "").lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = payload.credentials
    rsa_key = _get_rsa_key(token)
    if not rsa_key:
        raise HTTPException(status_code=401, detail="Invalid token (no matching JWKS key)")

    try:
        decoded = jwt.decode(
            token,
            rsa_key,
            algorithms=ALGORITHMS,
            audience=AUTH0_AUDIENCE,
            issuer=AUTH0_ISSUER,
        )
        return decoded
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

