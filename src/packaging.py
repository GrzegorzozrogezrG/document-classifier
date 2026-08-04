import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import py7zr


def _sanitize_filename(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name)
    return cleaned or "file.bin"


def create_7z_archive(items: List[Dict[str, Any]], archive_name: str) -> bytes:
    if not items:
        return b""

    archive_stem = Path(archive_name).stem or "classified_documents"

    with tempfile.TemporaryDirectory() as tmp_dir:
        export_root = Path(tmp_dir) / archive_stem
        export_root.mkdir(parents=True, exist_ok=True)

        for item in items:
            group_name = str(item.get("user_group", "general"))
            group_safe = _sanitize_filename(group_name)
            group_dir = export_root / group_safe
            group_dir.mkdir(parents=True, exist_ok=True)

            filename = _sanitize_filename(str(item.get("filename", "document.bin")))
            file_path = group_dir / filename

            with open(file_path, "wb") as handle:
                handle.write(item["raw_bytes"])

        archive_path = Path(tmp_dir) / (archive_stem + ".7z")
        with py7zr.SevenZipFile(archive_path, "w") as archive:
            archive.writeall(export_root, arcname=archive_stem)

        with open(archive_path, "rb") as handle:
            return handle.read()