"""GitHub API client for repository interactions."""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GitHubRepository:
    """GitHub repository information."""
    name: str
    full_name: str
    private: bool
    clone_url: str
    default_branch: str
    description: Optional[str] = None


@dataclass
class GitHubIssue:
    """GitHub issue information."""
    number: int
    title: str
    body: str
    state: str
    labels: List[str]
    assignees: List[str]


class GitHubClient:
    """Client for GitHub API interactions."""
    
    def __init__(self, token: Optional[str] = None, base_url: str = "https://api.github.com"):
        """Initialize GitHub client.
        
        Args:
            token: GitHub API token
            base_url: GitHub API base URL
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = base_url.rstrip("/")
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for GitHub integration")
        
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "causal-interface-gym/1.0"
            })
    
    def is_available(self) -> bool:
        """Check if GitHub client is properly configured.
        
        Returns:
            True if client can make API calls
        """
        return REQUESTS_AVAILABLE and self.token is not None
    
    def get_repository(self, owner: str, repo: str) -> Optional[GitHubRepository]:
        """Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository information or None
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            return GitHubRepository(
                name=data["name"],
                full_name=data["full_name"],
                private=data["private"],
                clone_url=data["clone_url"],
                default_branch=data["default_branch"],
                description=data.get("description")
            )
        except Exception as e:
            logger.error(f"Failed to get repository {owner}/{repo}: {e}")
            return None
    
    def create_issue(self, owner: str, repo: str, title: str, body: str, 
                    labels: Optional[List[str]] = None) -> Optional[GitHubIssue]:
        """Create a new issue.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: Issue title
            body: Issue body
            labels: Issue labels
            
        Returns:
            Created issue or None
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            payload = {
                "title": title,
                "body": body
            }
            
            if labels:
                payload["labels"] = labels
            
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return GitHubIssue(
                number=data["number"],
                title=data["title"],
                body=data["body"],
                state=data["state"],
                labels=[label["name"] for label in data["labels"]],
                assignees=[assignee["login"] for assignee in data["assignees"]]
            )
        except Exception as e:
            logger.error(f"Failed to create issue: {e}")
            return None
    
    def get_issues(self, owner: str, repo: str, state: str = "open", 
                  labels: Optional[str] = None) -> List[GitHubIssue]:
        """Get repository issues.
        
        Args:
            owner: Repository owner
            repo: Repository name
            state: Issue state (open, closed, all)
            labels: Comma-separated label names
            
        Returns:
            List of issues
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            params = {"state": state}
            
            if labels:
                params["labels"] = labels
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            issues = []
            for data in response.json():
                if "pull_request" not in data:  # Exclude pull requests
                    issues.append(GitHubIssue(
                        number=data["number"],
                        title=data["title"],
                        body=data["body"] or "",
                        state=data["state"],
                        labels=[label["name"] for label in data["labels"]],
                        assignees=[assignee["login"] for assignee in data["assignees"]]
                    ))
            
            return issues
        except Exception as e:
            logger.error(f"Failed to get issues: {e}")
            return []
    
    def create_pull_request(self, owner: str, repo: str, title: str, body: str,
                          head: str, base: str) -> Optional[Dict[str, Any]]:
        """Create a pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            title: PR title
            body: PR body
            head: Source branch
            base: Target branch
            
        Returns:
            Pull request data or None
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
            payload = {
                "title": title,
                "body": body,
                "head": head,
                "base": base
            }
            
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create pull request: {e}")
            return None
    
    def get_workflow_runs(self, owner: str, repo: str, workflow_id: str) -> List[Dict[str, Any]]:
        """Get workflow runs.
        
        Args:
            owner: Repository owner
            repo: Repository name
            workflow_id: Workflow ID or filename
            
        Returns:
            List of workflow runs
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs"
            response = self.session.get(url)
            response.raise_for_status()
            
            return response.json().get("workflow_runs", [])
        except Exception as e:
            logger.error(f"Failed to get workflow runs: {e}")
            return []
    
    def trigger_workflow(self, owner: str, repo: str, workflow_id: str, 
                        ref: str = "main", inputs: Optional[Dict[str, Any]] = None) -> bool:
        """Trigger a workflow dispatch.
        
        Args:
            owner: Repository owner
            repo: Repository name
            workflow_id: Workflow ID or filename
            ref: Git ref to run workflow on
            inputs: Workflow inputs
            
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"
            payload = {"ref": ref}
            
            if inputs:
                payload["inputs"] = inputs
            
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            return True
        except Exception as e:
            logger.error(f"Failed to trigger workflow: {e}")
            return False
    
    def update_repository_settings(self, owner: str, repo: str, 
                                 settings: Dict[str, Any]) -> bool:
        """Update repository settings.
        
        Args:
            owner: Repository owner
            repo: Repository name
            settings: Settings to update
            
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}"
            response = self.session.patch(url, json=settings)
            response.raise_for_status()
            
            return True
        except Exception as e:
            logger.error(f"Failed to update repository settings: {e}")
            return False
    
    def create_branch_protection(self, owner: str, repo: str, branch: str,
                               protection_rules: Dict[str, Any]) -> bool:
        """Create branch protection rules.
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name
            protection_rules: Protection configuration
            
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/branches/{branch}/protection"
            response = self.session.put(url, json=protection_rules)
            response.raise_for_status()
            
            return True
        except Exception as e:
            logger.error(f"Failed to create branch protection: {e}")
            return False