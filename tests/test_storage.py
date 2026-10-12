from pathlib import Path

import pytest

from support_ops.policies import load_policies, search_policies
from support_ops.storage import SupportStore


ROOT = Path(__file__).resolve().parents[1]


def seeded_store(tmp_path):
    store = SupportStore(tmp_path)
    store.initialize(ROOT / "data/orders.json")
    return store


def test_order_access_and_policy_sources(tmp_path):
    store = seeded_store(tmp_path)
    assert store.get_order("CUST-001", "ORD-1042")["status"] == "shipped"
    unavailable = {"found": False, "error": "order_unavailable"}
    assert store.get_order("CUST-001", "ORD-2099") == unavailable
    assert store.get_order("CUST-001", "ORD-0") == unavailable
    assert store.preview_cancel("CUST-001", "ORD-1042")["eligible"] is False
    policies = load_policies(ROOT / "data/policies.json")
    assert search_policies(policies, "politique de retour")["source_id"] == "POL-RETURNS"
    assert search_policies(policies, "unrelated question")["found"] is False


def test_approved_cancellation_is_idempotent(tmp_path):
    store = seeded_store(tmp_path)
    assert store.preview_cancel("CUST-001", "ORD-1050")["eligible"] is True
    first = store.cancel_order("CUST-001", "ORD-1050", "request-1")
    again = store.cancel_order("CUST-001", "ORD-1050", "request-1")
    assert first["outcome"] == again["outcome"] == "cancelled"
    assert first["replayed"] is False and again["replayed"] is True
    assert store.get_order("CUST-001", "ORD-1050")["status"] == "cancelled"
    assert store.preview_cancel("CUST-001", "ORD-1050")["reason"] == "status_cancelled"
    later = store.cancel_order("CUST-001", "ORD-1050", "request-later")
    assert later["outcome"] == "not_eligible"
    assert later["status"] == "cancelled"
    assert store.cancel_order("CUST-001", "ORD-1042", "request-2")["outcome"] == "not_eligible"
    assert store.cancel_order("CUST-001", "ORD-2099", "request-3") == {
        "found": False,
        "error": "order_unavailable",
    }
    with pytest.raises(ValueError):
        store.cancel_order("CUST-001", "ORD-1042", "request-1")


def test_reset_restores_seed_data(tmp_path):
    store = seeded_store(tmp_path)
    store.cancel_order("CUST-001", "ORD-1050", "request-1")
    store.reset_demo(ROOT / "data/orders.json")
    assert store.get_order("CUST-001", "ORD-1050")["status"] == "processing"
    assert store.cancel_order("CUST-001", "ORD-1050", "request-1")["replayed"] is False
