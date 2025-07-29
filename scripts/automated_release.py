#!/usr/bin/env python3
"""
Automated release script for causal-interface-gym.

This script automates the release process including:
- Version bumping
- Changelog generation
- Tag creation
- Package building
- PyPI publishing
"""

import argparse
import subprocess
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Version types
VERSION_TYPES = ["patch", "minor", "major"]


class ReleaseAutomator:
    """Handles automated releases for the project."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent
        
    def run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command with proper error handling."""
        if self.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=check,
            cwd=self.project_root
        )
        
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
            
        return result
    
    def get_current_version(self) -> str:
        """Get the current version from pyproject.toml."""
        pyproject_path = self.project_root / "pyproject.toml"
        
        with open(pyproject_path, 'r') as f:
            content = f.read()
            
        version_match = re.search(r'version = "([^"]+)"', content)
        if not version_match:
            raise ValueError("Could not find version in pyproject.toml")
            
        return version_match.group(1)
    
    def bump_version(self, version_type: str) -> str:
        """Bump version using semantic versioning."""
        current = self.get_current_version()
        major, minor, patch = map(int, current.split('.'))
        
        if version_type == "patch":
            patch += 1
        elif version_type == "minor":
            minor += 1
            patch = 0
        elif version_type == "major":
            major += 1
            minor = 0
            patch = 0
        else:
            raise ValueError(f"Invalid version type: {version_type}")
        
        new_version = f"{major}.{minor}.{patch}"
        
        # Update pyproject.toml
        pyproject_path = self.project_root / "pyproject.toml"
        with open(pyproject_path, 'r') as f:
            content = f.read()
        
        content = re.sub(
            r'version = "[^"]+"',
            f'version = "{new_version}"',
            content
        )
        
        if not self.dry_run:
            with open(pyproject_path, 'w') as f:
                f.write(content)
        
        # Update __init__.py
        init_path = self.project_root / "src" / "causal_interface_gym" / "__init__.py"
        with open(init_path, 'r') as f:
            content = f.read()
            
        content = re.sub(
            r'__version__ = "[^"]+"',
            f'__version__ = "{new_version}"',
            content
        )
        
        if not self.dry_run:
            with open(init_path, 'w') as f:
                f.write(content)
        
        return new_version
    
    def generate_changelog_entry(self, version: str) -> str:
        """Generate changelog entry from git commits."""
        # Get commits since last tag
        try:
            result = self.run_command([
                "git", "log", "--oneline", "--no-merges",
                f"v{self.get_current_version()}..HEAD"
            ], check=False)
            
            if result.returncode != 0:
                # No previous tag, get all commits
                result = self.run_command([
                    "git", "log", "--oneline", "--no-merges"
                ])
        except subprocess.CalledProcessError:
            result = self.run_command([
                "git", "log", "--oneline", "--no-merges", "-10"
            ])
        
        commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Categorize commits
        features = []
        fixes = []
        docs = []
        other = []
        
        for commit in commits:
            if commit.startswith(('feat:', 'feature:')):
                features.append(commit)
            elif commit.startswith(('fix:', 'bugfix:')):
                fixes.append(commit)
            elif commit.startswith(('docs:', 'doc:')):
                docs.append(commit)
            else:
                other.append(commit)
        
        # Generate changelog
        changelog = f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        if features:
            changelog += "### Added\n"
            for feat in features:
                changelog += f"- {feat.split(' ', 1)[1]}\n"
            changelog += "\n"
        
        if fixes:
            changelog += "### Fixed\n"
            for fix in fixes:
                changelog += f"- {fix.split(' ', 1)[1]}\n"
            changelog += "\n"
        
        if docs:
            changelog += "### Documentation\n"
            for doc in docs:
                changelog += f"- {doc.split(' ', 1)[1]}\n"
            changelog += "\n"
        
        if other:
            changelog += "### Other Changes\n"
            for change in other:
                changelog += f"- {change.split(' ', 1)[1]}\n"
            changelog += "\n"
        
        return changelog
    
    def update_changelog(self, version: str) -> None:
        """Update the CHANGELOG.md file."""
        changelog_path = self.project_root / "CHANGELOG.md"
        
        new_entry = self.generate_changelog_entry(version)
        
        if changelog_path.exists():
            with open(changelog_path, 'r') as f:
                existing_content = f.read()
        else:
            existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
        
        # Insert new entry after the header
        lines = existing_content.split('\n')
        header_end = 2  # After "# Changelog" and description
        
        new_content = '\n'.join(lines[:header_end + 1]) + '\n\n' + new_entry + '\n'.join(lines[header_end + 1:])
        
        if not self.dry_run:
            with open(changelog_path, 'w') as f:
                f.write(new_content)
    
    def run_quality_checks(self) -> bool:
        """Run quality checks before release."""
        print("🔍 Running quality checks...")
        
        checks = [
            (["python", "-m", "pytest", "tests/", "-v"], "Tests"),
            (["ruff", "check", "."], "Linting"),
            (["mypy", "."], "Type checking"),
            (["python", "scripts/validate_causal_accuracy.py"], "Causal accuracy"),
        ]
        
        for cmd, name in checks:
            print(f"Running {name}...")
            try:
                result = self.run_command(cmd)
                if result.returncode != 0:
                    print(f"❌ {name} failed!")
                    return False
                print(f"✅ {name} passed!")
            except subprocess.CalledProcessError:
                print(f"❌ {name} failed!")
                return False
        
        return True
    
    def build_package(self) -> None:
        """Build the package for distribution."""
        print("📦 Building package...")
        
        # Clean previous builds
        self.run_command(["rm", "-rf", "dist/", "build/"], check=False)
        
        # Build package
        self.run_command(["python", "-m", "build"])
    
    def create_git_tag(self, version: str) -> None:
        """Create and push git tag."""
        tag_name = f"v{version}"
        
        # Commit version changes
        self.run_command(["git", "add", "."])
        self.run_command([
            "git", "commit", "-m", f"release: bump version to {version}"
        ])
        
        # Create tag
        self.run_command([
            "git", "tag", "-a", tag_name, "-m", f"Release {version}"
        ])
        
        # Push changes and tag
        self.run_command(["git", "push", "origin", "main"])
        self.run_command(["git", "push", "origin", tag_name])
    
    def publish_to_pypi(self, test: bool = False) -> None:
        """Publish package to PyPI."""
        repository = "testpypi" if test else "pypi"
        print(f"🚀 Publishing to {repository}...")
        
        cmd = ["python", "-m", "twine", "upload"]
        if test:
            cmd.extend(["--repository", "testpypi"])
        cmd.append("dist/*")
        
        self.run_command(cmd)
    
    def create_github_release(self, version: str) -> None:
        """Create GitHub release."""
        print("🎉 Creating GitHub release...")
        
        changelog_path = self.project_root / "CHANGELOG.md"
        if changelog_path.exists():
            with open(changelog_path, 'r') as f:
                changelog_content = f.read()
            
            # Extract release notes for this version
            lines = changelog_content.split('\n')
            start_idx = None
            end_idx = None
            
            for i, line in enumerate(lines):
                if line.startswith(f"## [{version}]"):
                    start_idx = i
                elif start_idx is not None and line.startswith("## ["):
                    end_idx = i
                    break
            
            if start_idx is not None:
                if end_idx is None:
                    end_idx = len(lines)
                release_notes = '\n'.join(lines[start_idx:end_idx]).strip()
            else:
                release_notes = f"Release {version}"
        else:
            release_notes = f"Release {version}"
        
        # Create release using gh CLI
        self.run_command([
            "gh", "release", "create", f"v{version}",
            "--title", f"Release {version}",
            "--notes", release_notes,
            "dist/*"
        ])
    
    def perform_release(self, version_type: str, test_pypi: bool = False) -> None:
        """Perform the complete release process."""
        print(f"🚀 Starting {version_type} release...")
        
        if not self.dry_run and not self.run_quality_checks():
            print("❌ Quality checks failed. Aborting release.")
            sys.exit(1)
        
        # Bump version
        new_version = self.bump_version(version_type)
        print(f"📈 Version bumped to {new_version}")
        
        # Update changelog
        self.update_changelog(new_version)
        print("📝 Changelog updated")
        
        # Build package
        self.build_package()
        print("📦 Package built")
        
        if not self.dry_run:
            # Create git tag
            self.create_git_tag(new_version)
            print("🏷️  Git tag created")
            
            # Publish to PyPI
            self.publish_to_pypi(test=test_pypi)
            print("🚀 Published to PyPI")
            
            # Create GitHub release
            self.create_github_release(new_version)
            print("🎉 GitHub release created")
        
        print(f"✅ Release {new_version} completed successfully!")


def main():
    """Main release function."""
    parser = argparse.ArgumentParser(description="Automated release script")
    parser.add_argument(
        "version_type",
        choices=VERSION_TYPES,
        help="Type of version bump"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    parser.add_argument(
        "--test-pypi",
        action="store_true",
        help="Publish to test PyPI instead of production"
    )
    
    args = parser.parse_args()
    
    releaser = ReleaseAutomator(dry_run=args.dry_run)
    releaser.perform_release(args.version_type, test_pypi=args.test_pypi)


if __name__ == "__main__":
    main()