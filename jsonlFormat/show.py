import streamlit as st
import json
import pandas as pd

st.title("📄 JSONL Viewer")

uploaded_file = st.file_uploader("Upload a JSONL file", type=["jsonl"])

if uploaded_file is not None:
    # Decode file
    lines = uploaded_file.read().decode("utf-8").splitlines()
    
    # Parse JSONL
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            st.error(f"Invalid JSON line: {line}")
    
    # Show as raw JSON
    st.subheader("Raw JSON Objects")
    for rec in records:
        st.json(rec)
    
    # Convert to DataFrame if possible
    try:
        df = pd.DataFrame(records)
        st.subheader("Table View")
        st.dataframe(df)
    except Exception as e:
        st.warning(f"Could not convert to table: {e}")
