"""OAuth authentication provider."""

import os
import secrets
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import base64
import hashlib
import hmac

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OAuthConfig:
    """OAuth provider configuration."""
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    user_info_url: str
    scopes: List[str]
    redirect_uri: str


@dataclass
class OAuthToken:
    """OAuth token information."""
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if token is expired.
        
        Returns:
            True if token is expired
        """
        if not self.created_at:
            return True
        
        expiry_time = self.created_at + timedelta(seconds=self.expires_in - 60)  # 60s buffer
        return datetime.now() > expiry_time


@dataclass
class UserInfo:
    """User information from OAuth provider."""
    id: str
    email: str
    name: str
    avatar_url: Optional[str] = None
    provider: str = "unknown"


class OAuthProvider:
    """OAuth 2.0 authentication provider."""
    
    def __init__(self, provider_name: str, config: OAuthConfig):
        """Initialize OAuth provider.
        
        Args:
            provider_name: Name of the OAuth provider
            config: OAuth configuration
        """
        self.provider_name = provider_name
        self.config = config
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for OAuth")
        
        self.session = requests.Session()
    
    def get_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """Get authorization URL for OAuth flow.
        
        Args:
            state: Optional state parameter for security
            
        Returns:
            Tuple of (authorization_url, state)
        """
        if not state:
            state = secrets.token_urlsafe(32)
        
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "response_type": "code",
            "state": state
        }
        
        # Build URL
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        auth_url = f"{self.config.authorize_url}?{param_string}"
        
        return auth_url, state
    
    def exchange_code_for_token(self, code: str, state: Optional[str] = None) -> Optional[OAuthToken]:
        """Exchange authorization code for access token.
        
        Args:
            code: Authorization code from OAuth callback
            state: State parameter for verification
            
        Returns:
            OAuth token or None if failed
        """
        try:
            data = {
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "code": code,
                "redirect_uri": self.config.redirect_uri,
                "grant_type": "authorization_code"
            }
            
            response = self.session.post(
                self.config.token_url,
                data=data,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            
            token_data = response.json()
            
            return OAuthToken(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in", 3600),
                refresh_token=token_data.get("refresh_token"),
                scope=token_data.get("scope"),
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to exchange code for token: {e}")
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[OAuthToken]:
        """Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New OAuth token or None if failed
        """
        try:
            data = {
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token"
            }
            
            response = self.session.post(
                self.config.token_url,
                data=data,
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            
            token_data = response.json()
            
            return OAuthToken(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in", 3600),
                refresh_token=token_data.get("refresh_token", refresh_token),
                scope=token_data.get("scope"),
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            return None
    
    def get_user_info(self, token: OAuthToken) -> Optional[UserInfo]:
        """Get user information using access token.
        
        Args:
            token: OAuth access token
            
        Returns:
            User information or None if failed
        """
        try:
            headers = {
                "Authorization": f"{token.token_type} {token.access_token}",
                "Accept": "application/json"
            }
            
            response = self.session.get(self.config.user_info_url, headers=headers)
            response.raise_for_status()
            
            user_data = response.json()
            
            # Map common fields (provider-specific mapping may be needed)
            return UserInfo(
                id=str(user_data.get("id", user_data.get("sub", "unknown"))),
                email=user_data.get("email", ""),
                name=user_data.get("name", user_data.get("login", "Unknown")),
                avatar_url=user_data.get("avatar_url", user_data.get("picture")),
                provider=self.provider_name
            )
            
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None
    
    def validate_token(self, token: OAuthToken) -> bool:
        """Validate if token is still valid.
        
        Args:
            token: OAuth token to validate
            
        Returns:
            True if token is valid
        """
        if token.is_expired():
            return False
        
        # Try to get user info as validation
        user_info = self.get_user_info(token)
        return user_info is not None


class GitHubOAuthProvider(OAuthProvider):
    """GitHub-specific OAuth provider."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """Initialize GitHub OAuth provider.
        
        Args:
            client_id: GitHub OAuth app client ID
            client_secret: GitHub OAuth app client secret
            redirect_uri: OAuth redirect URI
        """
        config = OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            authorize_url="https://github.com/login/oauth/authorize",
            token_url="https://github.com/login/oauth/access_token",
            user_info_url="https://api.github.com/user",
            scopes=["user:email", "repo"],
            redirect_uri=redirect_uri
        )
        
        super().__init__("github", config)


class GoogleOAuthProvider(OAuthProvider):
    """Google-specific OAuth provider."""
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """Initialize Google OAuth provider.
        
        Args:
            client_id: Google OAuth client ID
            client_secret: Google OAuth client secret
            redirect_uri: OAuth redirect URI
        """
        config = OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            user_info_url="https://www.googleapis.com/oauth2/v2/userinfo",
            scopes=["openid", "email", "profile"],
            redirect_uri=redirect_uri
        )
        
        super().__init__("google", config)


def create_oauth_provider(provider_type: str, **kwargs) -> OAuthProvider:
    """Factory function to create OAuth providers.
    
    Args:
        provider_type: Type of OAuth provider
        **kwargs: Provider-specific configuration
        
    Returns:
        OAuth provider instance
    """
    providers = {
        "github": GitHubOAuthProvider,
        "google": GoogleOAuthProvider,
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unknown OAuth provider: {provider_type}")
    
    provider_class = providers[provider_type]
    return provider_class(**kwargs)