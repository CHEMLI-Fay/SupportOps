"""Validated synthetic customer and catalog records for the local demo database."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class DemoCustomer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    customer_id: str = Field(pattern=r"^CUST-[0-9]+$")
    name: str = Field(min_length=1)
    email: str = Field(pattern=r"^[^@]+@example\.com$")
    phone: str = Field(min_length=8)
    tier: str = Field(pattern=r"^(Standard|Silver|Gold)$")
    joined_at: str
    preferred_language: str
    address: str
    notes: str


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sku: str = Field(pattern=r"^SKU-[A-Z0-9-]+$")
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    price: float = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    stock: int = Field(ge=0)
    rating: float = Field(ge=0, le=5)
    reviews: int = Field(ge=0)
    image: str = Field(min_length=1)
    description: str = Field(min_length=1)


def _load_records(path: Path, model_type, label: str):
    try:
        records = TypeAdapter(list[model_type]).validate_python(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValidationError):
        raise ValueError(f"Invalid {label} fixture file.") from None
    return records


def load_customers(path: Path) -> dict[str, DemoCustomer]:
    records = _load_records(path, DemoCustomer, "customer")
    result = {record.customer_id: record for record in records}
    if len(result) != len(records):
        raise ValueError("Customer fixtures contain duplicate identifiers.")
    return result


def load_products(path: Path) -> dict[str, Product]:
    records = _load_records(path, Product, "product")
    result = {record.sku: record for record in records}
    if len(result) != len(records):
        raise ValueError("Product fixtures contain duplicate SKUs.")
    return result

