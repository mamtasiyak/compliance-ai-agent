from app.schemas.document import DocumentInput

test_document = DocumentInput(
    document_text=(
        "This is a compliance policy that explains how personal data must be handled within the organisation."
    )
)

# print(test_document)
# print(test_document.model_dump())
print(test_document.model_dump_json())