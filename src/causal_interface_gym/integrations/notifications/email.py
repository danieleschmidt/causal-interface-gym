"""Email notification client."""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EmailTemplate:
    """Email template configuration."""
    subject: str
    body_text: str
    body_html: Optional[str] = None
    attachments: Optional[List[str]] = None


class EmailClient:
    """SMTP email client for notifications."""
    
    def __init__(self, smtp_host: str, smtp_port: int = 587, 
                 username: Optional[str] = None, password: Optional[str] = None,
                 use_tls: bool = True):
        """Initialize email client.
        
        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            use_tls: Whether to use TLS encryption
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username or os.getenv("SMTP_USERNAME")
        self.password = password or os.getenv("SMTP_PASSWORD")
        self.use_tls = use_tls
        self.from_email = os.getenv("FROM_EMAIL", self.username)
        
        # Email templates
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, EmailTemplate]:
        """Load email templates.
        
        Returns:
            Dictionary of email templates
        """
        templates = {
            "experiment_complete": EmailTemplate(
                subject="Causal Experiment Complete - {experiment_id}",
                body_text="""
Your causal reasoning experiment {experiment_id} has completed successfully.

Results Summary:
- Agent: {agent_type}
- Interventions: {intervention_count}
- Causal Score: {causal_score:.3f}
- Duration: {duration}

View detailed results: {results_url}

Best regards,
Causal Interface Gym
""",
                body_html="""
<html>
<body>
<h2>Experiment Complete</h2>
<p>Your causal reasoning experiment <strong>{experiment_id}</strong> has completed successfully.</p>

<h3>Results Summary</h3>
<ul>
<li><strong>Agent:</strong> {agent_type}</li>
<li><strong>Interventions:</strong> {intervention_count}</li>
<li><strong>Causal Score:</strong> {causal_score:.3f}</li>
<li><strong>Duration:</strong> {duration}</li>
</ul>

<p><a href="{results_url}">View detailed results</a></p>

<p>Best regards,<br>Causal Interface Gym</p>
</body>
</html>
"""
            ),
            "experiment_failed": EmailTemplate(
                subject="Causal Experiment Failed - {experiment_id}",
                body_text="""
Your causal reasoning experiment {experiment_id} has failed.

Error Details:
{error_message}

Timestamp: {timestamp}
Agent: {agent_type}

Please check your configuration and try again.
If the problem persists, contact support.

Best regards,
Causal Interface Gym
"""
            ),
            "weekly_report": EmailTemplate(
                subject="Weekly Causal Reasoning Report",
                body_text="""
Weekly Report - Causal Interface Gym

Experiment Summary:
- Total Experiments: {total_experiments}
- Successful: {successful_experiments}
- Failed: {failed_experiments}
- Average Causal Score: {avg_causal_score:.3f}

Top Performing Agents:
{top_agents}

Most Common Issues:
{common_issues}

View full report: {report_url}

Best regards,
Causal Interface Gym
"""
            )
        }
        
        return templates
    
    def send_email(self, to_emails: List[str], subject: str, body_text: str,
                  body_html: Optional[str] = None, attachments: Optional[List[str]] = None) -> bool:
        """Send email.
        
        Args:
            to_emails: List of recipient email addresses
            subject: Email subject
            body_text: Plain text body
            body_html: HTML body (optional)
            attachments: List of file paths to attach
            
        Returns:
            True if email sent successfully
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            
            # Add text part
            text_part = MIMEText(body_text, 'plain', 'utf-8')
            msg.attach(text_part)
            
            # Add HTML part if provided
            if body_html:
                html_part = MIMEText(body_html, 'html', 'utf-8')
                msg.attach(html_part)
            
            # Add attachments
            if attachments:
                for attachment_path in attachments:
                    if Path(attachment_path).exists():
                        self._add_attachment(msg, attachment_path)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {to_emails}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _add_attachment(self, msg: MIMEMultipart, file_path: str) -> None:
        """Add file attachment to email.
        
        Args:
            msg: Email message object
            file_path: Path to file to attach
        """
        try:
            with open(file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {Path(file_path).name}'
            )
            
            msg.attach(part)
            
        except Exception as e:
            logger.warning(f"Failed to add attachment {file_path}: {e}")
    
    def send_template_email(self, template_name: str, to_emails: List[str],
                          template_vars: Dict[str, Any]) -> bool:
        """Send email using template.
        
        Args:
            template_name: Name of email template
            to_emails: List of recipient emails
            template_vars: Variables to substitute in template
            
        Returns:
            True if email sent successfully
        """
        if template_name not in self.templates:
            logger.error(f"Template '{template_name}' not found")
            return False
        
        template = self.templates[template_name]
        
        try:
            # Format template
            subject = template.subject.format(**template_vars)
            body_text = template.body_text.format(**template_vars)
            body_html = template.body_html.format(**template_vars) if template.body_html else None
            
            return self.send_email(
                to_emails=to_emails,
                subject=subject,
                body_text=body_text,
                body_html=body_html,
                attachments=template.attachments
            )
            
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return False
    
    def send_experiment_notification(self, to_emails: List[str], 
                                   experiment_data: Dict[str, Any]) -> bool:
        """Send experiment completion notification.
        
        Args:
            to_emails: Recipient emails
            experiment_data: Experiment results data
            
        Returns:
            True if successful
        """
        template_name = "experiment_complete" if experiment_data.get("success", True) else "experiment_failed"
        
        # Prepare template variables
        template_vars = {
            "experiment_id": experiment_data.get("experiment_id", "unknown"),
            "agent_type": experiment_data.get("agent_type", "unknown"),
            "intervention_count": len(experiment_data.get("interventions", [])),
            "causal_score": experiment_data.get("causal_score", 0.0),
            "duration": experiment_data.get("duration", "unknown"),
            "results_url": experiment_data.get("results_url", "#"),
            "error_message": experiment_data.get("error_message", "Unknown error"),
            "timestamp": experiment_data.get("timestamp", "unknown")
        }
        
        return self.send_template_email(template_name, to_emails, template_vars)
    
    def is_configured(self) -> bool:
        """Check if email client is properly configured.
        
        Returns:
            True if client can send emails
        """
        return bool(self.smtp_host and self.from_email)