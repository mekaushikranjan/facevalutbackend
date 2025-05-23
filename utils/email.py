import os
import emails
from emails.template import JinjaTemplate
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Email configuration
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAILS_FROM_EMAIL = os.getenv("EMAILS_FROM_EMAIL", "noreply@facevault.com")
EMAILS_FROM_NAME = os.getenv("EMAILS_FROM_NAME", "FaceVault")

# Email templates directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "email"

def send_email(
    email_to: str,
    subject: str,
    template_name: str,
    template_data: dict,
    html_template: Optional[str] = None
) -> bool:
    """
    Send an email using the specified template or custom HTML.
    
    Args:
        email_to: Recipient email address
        subject: Email subject
        template_name: Name of the template file (without extension)
        template_data: Dictionary of template variables
        html_template: Optional custom HTML template string
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    try:
        if html_template:
            template = JinjaTemplate(html_template)
        else:
            template_path = TEMPLATES_DIR / f"{template_name}.html"
            template = JinjaTemplate(template_path.read_text())

        message = emails.Message(
            subject=subject,
            html=template.render(**template_data),
            mail_from=(EMAILS_FROM_NAME, EMAILS_FROM_EMAIL),
        )

        response = message.send(
            to=email_to,
            render=template_data,
            smtp={
                "host": SMTP_HOST,
                "port": SMTP_PORT,
                "user": SMTP_USER,
                "password": SMTP_PASSWORD,
                "tls": True,
            },
        )
        
        if response.status_code not in [250, 200, 201, 202]:
            logger.error(f"Failed to send email: {response.status_code} - {response.text}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        return False

def send_otp_email(email_to: str, otp_code: str) -> bool:
    """Send OTP verification email."""
    return send_email(
        email_to=email_to,
        subject="Your FaceVault Verification Code",
        template_name="otp",
        template_data={"otp_code": otp_code},
    )

def send_reset_password_email(email_to: str, reset_url: str) -> bool:
    """Send password reset email."""
    return send_email(
        email_to=email_to,
        subject="Reset Your FaceVault Password",
        template_name="reset_password",
        template_data={"reset_url": reset_url},
    ) 