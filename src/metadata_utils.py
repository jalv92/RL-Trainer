"""Utility helpers for persisting and loading training metadata."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def metadata_path(resource_path: str) -> str:
    """Return the metadata path associated with a resource file."""
    return f"{resource_path}.meta.json"


def write_metadata(resource_path: str, metadata: Dict[str, Any]) -> str:
    """Write metadata next to a resource file.

    Args:
        resource_path: Path to the model or artifact the metadata describes.
        metadata: JSON-serializable dictionary with metadata fields.

    Returns:
        The path to the metadata file written.
    """
    meta_path = metadata_path(resource_path)
    payload = dict(metadata) if metadata is not None else {}
    payload.setdefault("resource", os.path.abspath(resource_path))
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return meta_path


def read_metadata(resource_path: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Read metadata for a resource if it exists."""
    meta_path = metadata_path(resource_path)
    if not os.path.exists(meta_path):
        return default
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def require_metadata_field(metadata: Dict[str, Any], field: str) -> Any:
    """Return a required metadata field or raise a ValueError."""
    if metadata is None or field not in metadata:
        raise ValueError(f"Missing required metadata field '{field}'")
    return metadata[field]
