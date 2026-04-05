import csv
import io
import json

import requests
import streamlit as st

st.title("Compliance Obligation Extractor")

# Initialise result storage on first load
if "result" not in st.session_state:
    st.session_state.result = None

RISK_COLORS = {
    "Critical": "🔴",
    "High":     "🟠",
    "Medium":   "🟡",
    "Low":      "🟢",
}

document_text = st.text_area(
    label="Paste your document here",
    height=300,
    placeholder="e.g. All staff must complete annual GDPR training...",
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Extract Obligations", disabled=len(document_text.strip()) < 50):
        with st.spinner("Extracting obligations..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/extract",
                    json={"document_text": document_text},
                    timeout=30,
                )
                if response.status_code == 200:
                    st.session_state.result = response.json()
                else:
                    st.error(f"API error {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Make sure the FastAPI server is running.")
            except requests.exceptions.Timeout:
                st.error("The API took too long to respond. Try again.")

with col2:
    if st.button("Clear", disabled=st.session_state.result is None):
        st.session_state.result = None
        st.rerun()

# Display results
if st.session_state.result:
    data = st.session_state.result

    st.success(f"Extraction complete. Found {data['total_obligations']} obligations.")
    st.subheader("Document Summary")
    st.write(data["document_summary"])

    st.subheader(f"Obligations Found: {data['total_obligations']}")

    for i, ob in enumerate(data["obligations"], start=1):
        risk = ob["risk_level"]
        icon = RISK_COLORS.get(risk, "⚪")
        with st.expander(f"{i}. {ob['obligation']}  —  {icon} {risk}"):
            st.write(f"**Description:** {ob['description']}")
            st.write(f"**Source text:** _{ob['source_text']}_")
            if data["total_obligations"] == 0:
                st.warning("No obligations found in the document.")
            if ob.get("deadline"):
                st.write(f"**Deadline:** {ob['deadline']}")
            if ob.get("responsible_entity"):
                st.write(f"**Responsible:** {ob['responsible_entity']}")
            if ob.get("section_reference"):
                st.write(f"**Section:** {ob['section_reference']}")
            st.write(f"**Confidence:** {round(ob['confidence'] * 100)}%")

    # Export
    st.divider()
    export_col1, export_col2 = st.columns(2)

    with export_col1:
        st.download_button(
            label="Download JSON",
            data=json.dumps(data, indent=2),
            file_name="obligations.json",
            mime="application/json",
        )

    with export_col2:
        buffer = io.StringIO()
        fields = ["obligation", "risk_level", "description", "deadline",
                  "responsible_entity", "source_text", "section_reference", "confidence"]
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data["obligations"])
        st.download_button(
            label="Download CSV",
            data=buffer.getvalue(),
            file_name="obligations.csv",
            mime="text/csv",
        )
