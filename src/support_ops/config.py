"""Load validated project settings without exposing credentials."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    reasoning_effort: Literal["low", "medium", "high"]
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    max_retries: int = Field(ge=0, le=2)


class FileSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    orders: str = Field(min_length=1)


class DemoSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str = Field(pattern=r"^CUST-[0-9]+$")
    order_id: str = Field(pattern=r"^ORD-[0-9]+$")
    question: str = Field(min_length=1)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm: LLMSettings
    files: FileSettings
    demo: DemoSettings


def find_project_root(start: Path | None = None) -> Path:
    """Find the checkout from its root, notebook directory, or a child folder."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src/support_ops").is_dir() and (
            candidate / "config/settings.toml"
        ).is_file():
            return candidate
    raise ValueError("Open the notebook or terminal inside the AgentOps checkout.")


def project_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if Path(relative_path).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Configured files must stay inside the project directory.")
    if not path.is_file():
        raise ValueError(f"Required project file is missing: {relative_path}")
    return path


def load_settings(root: Path) -> Settings:
    try:
        with project_file(root, "config/settings.toml").open("rb") as stream:
            settings = Settings.model_validate(tomllib.load(stream))
    except (tomllib.TOMLDecodeError, ValidationError):
        raise ValueError("Invalid config/settings.toml: check field names and values.") from None
    project_file(root, settings.files.prompt)
    project_file(root, settings.files.orders)
    return settings


def load_prompt(root: Path, settings: Settings) -> str:
    prompt = project_file(root, settings.files.prompt).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("The support prompt must not be empty.")
    return prompt


def load_api_key(root: Path) -> SecretStr:
    """Existing environment variables take precedence over the local .env file."""
    load_dotenv(root / ".env", override=False)
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key or key == "replace_with_your_groq_api_key":
        raise ValueError("Set GROQ_API_KEY in your local .env file before the API test.")
    return SecretStr(key)
