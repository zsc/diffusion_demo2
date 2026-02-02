"""I/O utilities for images, paths, and JSON."""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Union

from PIL import Image


def to_base64_png(image: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def from_base64_png(base64_str: str) -> Image.Image:
    """Convert base64 PNG string to PIL Image."""
    data = base64.b64decode(base64_str)
    return Image.open(BytesIO(data))


def save_json(path: Union[str, Path], data: Any) -> None:
    """Save data as JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Get project root directory (where this file is located)."""
    return Path(__file__).parent.parent.parent.parent


def get_relative_path(path: Union[str, Path], base: Union[str, Path] = None) -> str:
    """Get relative path from base (default: project root)."""
    if base is None:
        base = get_project_root()
    return str(Path(path).relative_to(Path(base)))
