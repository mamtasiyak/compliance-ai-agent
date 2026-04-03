from pydantic import BaseModel, Field, field_validator

class DocumentInput(BaseModel):
    """
    Input payload submitted by the caller for compliance extraction.

    Wraps the raw document text that the AI pipeline will analyse.
    Validation ensures the document is non-empty and within processable limits.
    """
    document_text: str = Field(...,min_length=50, max_length=500000)

    @field_validator("document_text")
    @classmethod
    def document_text_must_not_be_blank(cls, value: str) -> str:
        """Reject documents that contain only whitespace."""
        if not value.strip():
            raise ValueError(
                "'document_text' must contain readable content, not only whitespace."
            )
        return value.strip()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "document_text": (
                    "1. Data Retention Policy\n\n"
                    "Personal data shall be kept in a form which permits identification "
                    "of data subjects for no longer than is necessary for the purposes "
                    "for which the personal data are processed..."
                )
            }
        }
    }
    
#     document_text: str = Field(
#     ...,
#     min_length=50,
#     max_length=500_000,
#     description=(
#         "The full text content of the document to be analysed. "
#         "Plain text is preferred; pre-processed PDF or DOCX extractions are accepted. "
#         "Must be between 50 and 500,000 characters."
#     ),
#     examples=["[Full document text here...]"],
# )