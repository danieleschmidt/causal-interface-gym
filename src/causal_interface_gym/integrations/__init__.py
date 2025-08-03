"""External service integrations for causal interface gym."""

from .github import GitHubClient, GitHubWebhookHandler
from .notifications import (
    EmailClient,
    SlackClient,
    NotificationManager,
)
from .auth import (
    OAuthProvider,
    JWTManager,
    AuthenticationManager,
)
from .monitoring import (
    MetricsCollector,
    AlertManager,
    HealthChecker,
)

__all__ = [
    "GitHubClient",
    "GitHubWebhookHandler",
    "EmailClient",
    "SlackClient",
    "NotificationManager",
    "OAuthProvider",
    "JWTManager",
    "AuthenticationManager",
    "MetricsCollector",
    "AlertManager",
    "HealthChecker",
]