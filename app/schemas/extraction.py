from pydantic import BaseModel, Field, model_validator
from app.schemas.compliance import ComplianceObligation

class ExtractionResult(BaseModel):
    """
    Aggregated output of the compliance extraction pipeline for a single document.

    Contains a high-level summary of the document, the count of obligations
    found, and the full list of extracted ComplianceObligation objects.
    """
    document_summary: str = Field(..., min_length=10, max_length=1000)
    total_obligations: int = Field(..., ge=0)
    obligations: list[ComplianceObligation] = Field(default_factory=list)

    @model_validator(mode="after")
    def total_must_match_obligations_length(self) -> "ExtractionResult":
        """Ensure total_obligations is consistent with the obligations list."""
        if self.total_obligations != len(self.obligations):
            raise ValueError(
                f"'total_obligations' ({self.total_obligations}) does not match "
                f"the number of items in 'obligations' ({len(self.obligations)}). "
                "These values must be equal."
            )
        return self
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "document_summary": (
                    "Internal data governance policy covering GDPR obligations "
                    "for EU customer data."
                ),
                "total_obligations": 1,
                "obligations": [ComplianceObligation.model_config["json_schema_extra"]["example"]],
            }
        }
    }   

    # document_summary: str = Field(
    #     ...,
    #     min_length=10,
    #     max_length=1000,
    #     description=(
    #         "A concise summary of the analysed document, capturing its purpose, "
    #         "regulatory context, and primary subject matter."
    #     ),
    #     examples=[
    #         "Internal data governance policy outlining GDPR obligations for EU "
    #         "customer data handling, retention, and breach response procedures."
    #     ],
    # )
    # total_obligations: int = Field(
    #     ...,
    #     ge=0,
    #     description=(
    #         "The total number of distinct compliance obligations extracted. "
    #         "Must equal the length of the 'obligations' list."
    #     ),
    #     examples=[5],
    # )
    # obligations: list[ComplianceObligation] = Field(
    #     default_factory=list,
    #     description="Ordered list of all ComplianceObligation objects extracted from the document.",
    # )