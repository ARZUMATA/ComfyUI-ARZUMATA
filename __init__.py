import os
import importlib
from pathlib import Path

# Initialize mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Current directory (where __init__.py is)
current_dir = Path(__file__).parent

# Nodes directory
nodes_dir = current_dir / "nodes"

# Dynamically import all Python files in nodes folder
for py_file in nodes_dir.glob("*.py"):
    if py_file.name == "__init__.py":
        continue  # Skip __init__.py

    # Module name without .py
    module_name = f".nodes.{py_file.stem}"

    # Import module
    try:
        module = importlib.import_module(module_name, package=__package__ or __name__.split('.')[0])

        # Merge NODE_CLASS_MAPPINGS if present
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)

        # Merge NODE_DISPLAY_NAME_MAPPINGS if present
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]