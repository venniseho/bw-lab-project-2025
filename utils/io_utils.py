"""
io_utils.py
--------------------------------------------------------------
Input/Output and Path Management Utilities.

Standardizes how the pipeline reads and writes manifest files 
(JSONL) and resolves file paths across different stages.

Main Functions:
    - read_manifest: Loads a manifest.jsonl into a list of dicts.
    - write_manifest_line: Appends a single result to a JSONL file.
    - resolve_path: Ensures paths are absolute and user-expanded.
--------------------------------------------------------------
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

def read_manifest(manifest_path: Path) -> List[Dict]:
    """
    Reads a JSONL manifest file.
    
    Args:
        manifest_path: Path to the .jsonl file.
    Returns:
        A list of dictionaries containing image/mask paths.
    """
    rows = []
    if not manifest_path.exists():
        return rows
        
    with open(manifest_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolves a path string into a Path object. If the path is relative,
    it is resolved against base_dir.
    """
    p = Path(path_str).expanduser()
    if not p.is_absolute() and base_dir:
        return (base_dir / p).resolve()
    return p.resolve()