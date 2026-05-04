"""Pydantic request modelleri."""
from pydantic import BaseModel, Field
from typing import Optional


class EmailRequest(BaseModel):
    email_id: Optional[str] = None
    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1, max_length=50000)
    sender: Optional[str] = None
    source: str = Field(default="api", pattern="^(api|dataset|upload)$")

    class Config:
        json_schema_extra = {
            "example": {
                "subject": "URGENT: Verify your account immediately",
                "body": "Dear Customer, Your account will be suspended. Click here: http://paypa1.com/verify",
                "sender": "security@paypa1.com"
            }
        }


class BatchEmailRequest(BaseModel):
    emails: list[EmailRequest] = Field(..., max_length=100)
