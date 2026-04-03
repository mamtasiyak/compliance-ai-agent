from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator

class RiskLevel(str, Enum):
    """Enumeration of possible risk severity levels for a compliance obligation."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class ComplianceObligation(BaseModel):
    """
    Represents a single compliance obligation extracted from a document.

    Each obligation captures a discrete rule or requirement, its context, 
    risk severity, ownership and the model's confidence in the extraction.
    """

    obligation: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10)
    risk_level: RiskLevel = Field(...)
    deadline: Optional[str] = Field(default=None, max_length=300)
    responsible_entity: Optional[str] = Field(default=None, max_length=200)
    source_text: str = Field(..., min_length=5)
    section_reference: Optional[str] = Field(default=None, max_length=100)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("obligation")
    @classmethod
    def obligation_must_not_be_blank(cls, value:str) -> str:
        """Reject obligations that are whitespace-only."""
        if not value.strip():
            raise ValueError("Obligation must not be blank or whitespace.")
        return value.strip()
    
    @field_validator("source_text")
    @classmethod
    def source_text_must_not_be_blank(cls, value: str) -> str:
        """Reject source_text that is whitespace-only."""
        if not value.strip():
            raise ValueError("'source_text' must not be blank or whitespace-only.")
        return value.strip()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "obligation": "72-Hour Breach Notification",
                "description": (
                    "The organisation must notify the supervisory authority of a "
                    "personal data breach within 72 hours of becoming aware of it, "
                    "unless the breach is unlikely to result in a risk to individuals."
                ),
                "risk_level": "critical",
                "deadline": "Within 72 hours of breach discovery",
                "responsible_entity": "Data Protection Officer",
                "source_text": (
                    "The controller shall without undue delay and, where feasible, "
                    "not later than 72 hours after having become aware of it, notify "
                    "the personal data breach to the supervisory authority."
                ),
                "section_reference": "Article 33(1)",
                "confidence": 0.97,
            }
        }
    }
    
    # obligation: str = Field(
    #     ...,
    #     min_length=3,
    #     max_length=200,
    #     description="A short, human-readable title summarising the compliance rule.",
    #     examples=["Annual Data Retention Audit"],
    # )
    # description: str = Field(
    #     ...,
    #     min_length=10,
    #     description=(
    #         "A detailed explanation of the obligation, including scope, "
    #         "requirements, and any conditions that trigger it."
    #     ),
    #     examples=[
    #         "All customer PII must be reviewed annually and purged if retention "
    #         "criteria are no longer met, per GDPR Article 5(1)(e)."
    #     ],
    # )
    # risk_level: RiskLevel = Field(
    #     ...,
    #     description=(
    #         "Severity of non-compliance risk. "
    #         "One of: 'low', 'medium', 'high', 'critical'."
    #     ),
    #     examples=[RiskLevel.HIGH],
    # )
    # deadline: Optional[str] = Field(
    #     default=None,
    #     max_length=300,
    #     description=(
    #         "Human-readable description of the time requirement, e.g. "
    #         "'Within 72 hours of breach discovery' or 'Annually by 31 March'."
    #     ),
    #     examples=["Within 72 hours of a confirmed data breach"],
    # )
    # responsible_entity: Optional[str] = Field(
    #     default=None,
    #     max_length=200,
    #     description=(
    #         "The team, role, or individual accountable for fulfilling this obligation, "
    #         "e.g. 'Data Protection Officer' or 'Legal & Compliance Team'."
    #     ),
    #     examples=["Data Protection Officer"],
    # )
    # source_text: str = Field(
    #     ...,
    #     min_length=5,
    #     description=(
    #         "The verbatim excerpt from the source document that gives rise to "
    #         "this obligation. Used for audit traceability."
    #     ),
    #     examples=[
    #         "Personal data shall be kept in a form which permits identification "
    #         "of data subjects for no longer than is necessary."
    #     ],
    # )
    # section_reference: Optional[str] = Field(
    #     default=None,
    #     max_length=100,
    #     description=(
    #         "Reference to the location of the source text within the document, "
    #         "e.g. 'Section 4.2', 'Page 17', or 'Clause 8(b)'."
    #     ),
    #     examples=["Section 4.2 — Data Retention"],
    # )
    # confidence: float = Field(
    #     ...,
    #     ge=0.0,
    #     le=1.0,
    #     description=(
    #         "Model confidence score for this extraction, expressed as a float "
    #         "between 0.0 (no confidence) and 1.0 (certain)."
    #     ),
    #     examples=[0.92],
    # )
