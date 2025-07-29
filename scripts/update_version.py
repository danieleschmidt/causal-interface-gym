#!/usr/bin/env python3
"""Update version in project files."""

import sys
import re
from pathlib import Path

def update_version(new_version: str):
    """Update version in all relevant files."""
    
    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        content = re.sub(
            r'version = "[^"]*"',
            f'version = "{new_version}"',
            content
        )
        pyproject_path.write_text(content)
        print(f"Updated version in {pyproject_path}")
    
    # Update __init__.py
    init_path = Path("src/causal_interface_gym/__init__.py")
    if init_path.exists():
        content = init_path.read_text()
        content = re.sub(
            r'__version__ = "[^"]*"',
            f'__version__ = "{new_version}"',
            content
        )
        init_path.write_text(content)
        print(f"Updated version in {init_path}")
    
    # Update documentation conf.py if it exists
    docs_conf = Path("docs/conf.py")
    if docs_conf.exists():
        content = docs_conf.read_text()
        content = re.sub(
            r"version = '[^']*'",
            f"version = '{new_version}'",
            content
        )
        content = re.sub(
            r"release = '[^']*'",
            f"release = '{new_version}'",
            content
        )
        docs_conf.write_text(content)
        print(f"Updated version in {docs_conf}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <new_version>")
        sys.exit(1)
    
    new_version = sys.argv[1]
    update_version(new_version)
    print(f"Version updated to {new_version}")