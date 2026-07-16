# src/api_client.py

import time
import os
from typing import Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv

DEFAULT_CACHE_TTL_SECONDS = 600
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL")

_CACHE: Dict[Tuple[str, Tuple[Tuple[str, object], ...]], Tuple[float, Dict]] = {}


def get_json(
    path: str,
    params: Optional[Dict] = None,
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    timeout: int = 10,
) -> Dict:
    """Fetch JSON from the Nailify API with a small in-process TTL cache."""
    params = params or {}
    cache_key = (path, tuple(sorted(params.items())))
    now = time.time()

    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] < ttl_seconds:
        return cached[1]

    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    _CACHE[cache_key] = (now, payload)
    return payload


def fetch_paginated(path: str, page_size: int = 50, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS) -> List[Dict]:
    """Fetch a paginated Nailify API collection with 10-minute response caching."""
    items: List[Dict] = []
    page = 1

    while True:
        payload = get_json(
            path,
            params={"pageNumber": page, "pageSize": page_size},
            ttl_seconds=ttl_seconds,
        )
        if not payload.get("isSucceeded"):
            break

        data = payload.get("data") or {}
        items.extend(data.get("items") or [])

        if not (data.get("metaData") or {}).get("hasNext"):
            break
        page += 1

    return items


def fetch_products(ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS) -> Dict[str, List[Dict]]:
    """Fetch all product lists used by the RAG engine."""
    return {
        "shapes": fetch_paginated("/NailShapes", ttl_seconds=ttl_seconds),
        "surfaces": fetch_paginated("/NailSurfaces", ttl_seconds=ttl_seconds),
        "components": fetch_paginated("/Components", ttl_seconds=ttl_seconds),
    }


def fetch_customer(ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS) -> Dict:
    """Fetch the current customer profile from the Nailify API."""
    payload = get_json("/Profile/customers", ttl_seconds=ttl_seconds)
    if not payload.get("isSucceeded"):
        raise RuntimeError(payload.get("message", "Customer API request failed"))
    return payload.get("data") or {}


def fetch_customer_by_id(customer_id: str, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS) -> Dict:
    """Fetch one customer profile by user/customer id."""
    payload = get_json(f"/Users/customers/{customer_id}", ttl_seconds=ttl_seconds)
    if not payload.get("isSucceeded"):
        raise RuntimeError(payload.get("message", "Customer API request failed"))
    return payload.get("data") or {}
