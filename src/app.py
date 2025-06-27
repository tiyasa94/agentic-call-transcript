import streamlit as st
import pandas as pd
import os
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from main import pipeline, clean_output_folder, clean_and_convert_to_json


def load_json_for_customer(output_dir, customer_id, filename):
    path = os.path.join(output_dir, customer_id, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def run_pipeline_for_customer(customer_row, call_rows, progress_area):
    try:
        customer_id = customer_row['customer_id']
        progress_area.markdown(f"### 🤖 Agents are running...\n**Customer_ID:** `{customer_id}`")

        customer_data = customer_row.to_dict()
        customer_data['customer_id'] = customer_id

        call_rows['call_date'] = pd.to_datetime(call_rows['call_date'], errors='coerce')
        call_data = call_rows.sort_values('call_date').iloc[-1]

        context_summary = (
            f"The following call took place on {call_data['call_date']} at {call_data['call_time']} "
            f"between customer {call_data['customer_id']} and agent {call_data['agent_id']}. "
            f"The call lasted for {call_data['call_duration']} minutes.\n\n"
        )
        full_transcript = context_summary + call_data['transcript']

        pipeline(full_transcript, customer_data, customer_id)

        output_dir = "./output"
        customer_profile = load_json_for_customer(output_dir, customer_id, "customer_profile.json")
        # call_transcript = load_json_for_customer(output_dir, customer_id, "call_transcript.json")
        sentiment = load_json_for_customer(output_dir, customer_id, "sentiment.json")
        rootcause = load_json_for_customer(output_dir, customer_id, "rootcause.json")
        churn = load_json_for_customer(output_dir, customer_id, "churn_risk.json")
        retention = load_json_for_customer(output_dir, customer_id, "retention_recommender.json")

        progress_area.markdown("---")
        progress_area.markdown("### Customer Profile")
        progress_area.markdown(customer_profile)

        progress_area.markdown("### Call Transcript")
        progress_area.markdown(full_transcript)

        progress_area.markdown("### ✅ Step 1: Sentiment Analysis")
        progress_area.json(sentiment)

        progress_area.markdown("### ✅ Step 2: Root Cause Analysis")
        progress_area.json(rootcause)

        progress_area.markdown("### ✅ Step 3: Churn Prediction")
        progress_area.json(churn)

        progress_area.markdown("### ✅ Step 4: Retention Strategy Recommendation")
        progress_area.json(retention)

        progress_area.success(f"✅ All steps completed for {customer_id}")

        return {
            **customer_data,
            "intent": sentiment.get("intent", ""),
            "keywords": ", ".join(sentiment.get("keywords", [])),
            "sentiment": sentiment.get("sentiment", ""),
            "tone": sentiment.get("tone", ""),
            "churn_score": churn.get("churn_risk_score", ""),
            "churn_label": churn.get("churn_risk_label", ""),
            "root_cause": rootcause.get("root_cause", ""),
            "recommendation": retention.get("retention_strategy", "")
        }

    except Exception as e:
        error_msg = f"❌ Error processing {customer_row['customer_id']}: {e}"
        progress_area.error(error_msg)
        return None

# --- Streamlit App ---
st.title("Churn Analysis Dashboard with Causes & Recommendations")

st.sidebar.header("📂 Upload Files")
customer_file = st.sidebar.file_uploader("Upload Customer Data CSV", type=["csv"])
transcript_file = st.sidebar.file_uploader("Upload Call Transcript CSV", type=["csv"])

if customer_file and transcript_file:
    customer_df = pd.read_csv(customer_file)
    transcript_df = pd.read_csv(transcript_file)

    st.sidebar.header("🔍 Filters")
    country = st.sidebar.selectbox("Country", options=sorted(customer_df['Country'].dropna().unique()), index=customer_df['Country'].dropna().unique().tolist().index('India'))
    state = st.sidebar.selectbox("State", options=sorted(customer_df['State'].dropna().unique()), index=customer_df['State'].dropna().unique().tolist().index('Karnataka'))
    city = st.sidebar.selectbox("City", options=sorted(customer_df['City'].dropna().unique()), index=customer_df['City'].dropna().unique().tolist().index('Bengaluru'))
    contract = st.sidebar.selectbox("Contract Type", options=sorted(customer_df['contract_type'].dropna().unique()))

    filtered_df = customer_df[
        (customer_df['Country'] == country) &
        (customer_df['State'] == state) &
        (customer_df['City'] == city) &
        (customer_df['contract_type'] == contract)
    ]

    st.subheader("Filtered Customers")
    st.dataframe(filtered_df)


    if st.button("Run Pipeline"):
        output_rows = []
        clean_output_folder()

        for _, customer_row in filtered_df.iterrows():
            cust_id = customer_row['customer_id']
            customer_transcripts = transcript_df[transcript_df['customer_id'] == cust_id]
            progress_area = st.expander(f"Processing: {cust_id}", expanded=True)
            with progress_area:
                result = run_pipeline_for_customer(customer_row, customer_transcripts, progress_area)
                if result:
                    output_rows.append(result)

        if output_rows:
            result_df = pd.DataFrame(output_rows)
            st.success("✅ Pipeline completed for all filtered customers")
            st.download_button("📥 Download Result CSV", result_df.to_csv(index=False), file_name="churn_dashboard_output.csv")
            st.dataframe(result_df)

else:
    st.info("👆 Please upload both Customer and Call Transcript CSV files to begin.")
