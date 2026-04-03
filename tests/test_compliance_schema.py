from app.schemas.compliance import ComplianceObligation, RiskLevel

test_obligation = ComplianceObligation(
    obligation="72 hour breach notification",
    description="Company must report personal data breaches within 72 hours.",
    risk_level=RiskLevel.CRITICAL,
    deadline="Within 72 hours",
    responsible_entity="Data Protection Officer",
    source_text="The controller shall notify the supervisory authority within 72 hours.",
    section_reference="Article 33",
    confidence=0.5
)

# print(test_obligation)
# print(test_obligation.model_dump())
print(test_obligation.model_dump_json())

