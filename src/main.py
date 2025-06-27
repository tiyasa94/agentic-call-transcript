from crewai import Crew
import json
from dotenv import load_dotenv
import re
from datetime import datetime
from agents.customer_profile_summarizer import CustomerProfileSummarizerCrew 
import pandas as pd
from agents.sentiment import SentimentToneAnalyzerCrew
from agents.root_cause_analysis import RootCauseAnalysisCrew
from agents.churn_prediction import ChurnPredictionCrew
from agents.retention_recommender import RetentionStrategyRecommenderCrew
import os
load_dotenv("/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/.env", override=True)

# Instantiate Crews
sentiment_crew = SentimentToneAnalyzerCrew()
root_cause_crew = RootCauseAnalysisCrew()
churn_crew = ChurnPredictionCrew()
retention_crew = RetentionStrategyRecommenderCrew()

def clean_and_convert_to_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        print("No valid JSON found in the input text.")
        return None

    raw = match.group(0)

    # Fix unquoted list items for keys like 'root_cause' and 'real_time_guidance'
    fixed = re.sub(r'(\[\s*)(\* .+?)(\s*[,|\]])', lambda m: f'["{m.group(2).strip()}"]' if m.group(3) == "]" else f'"{m.group(2).strip()}",', raw)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print("⛔ Raw output with error:\n", raw)
        return None

def log_to_file(title, content):
    with open("pipeline_output_log.txt", "a") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"{datetime.now()} | {title}\n")
        f.write("="*60 + "\n")
        f.write(json.dumps(content, indent=2) if isinstance(content, dict) else str(content))
        f.write("\n\n")


def save_json_output(filename, data, customer_id):
    output_dir = f"/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/output/{customer_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"✅ Saved: {output_path}")

def clean_output_folder():
    output_dir = "/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/output"
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"🧹 Removed: {filename}")
        except Exception as e:
            print(f"⚠️ Failed to delete {filename}: {e}")


def pipeline(call_transcript, customer_profile, target_customer_id):
    
    save_json_output("customer_profile.json", customer_profile, target_customer_id)
    # save_json_output("call_transcript.json", call_transcript, target_customer_id)

    rca_json = {}  

    # Step 1: Sentiment Analysis
    sentiment_inputs = {
            "call_transcript": call_transcript,
            "customer_profile": customer_profile
        }
    sentiment_output = sentiment_crew.crew().kickoff(inputs=sentiment_inputs)
    sentiment_json = clean_and_convert_to_json(sentiment_output.raw)
    # sentiment_json["customer_id"] = target_customer_id
    save_json_output("sentiment.json", sentiment_json, target_customer_id)

    if not sentiment_json or sentiment_json["sentiment"] == "not sufficient information":
        print("\nStopping pipeline: Insufficient information in transcript.")
        return

    sentiment_label = sentiment_json["sentiment"].lower()
    # print(f"\nDetected Sentiment: {sentiment_label}")

    # Step 2: If POSITIVE → END
    if sentiment_label == "positive":
        print("\nSentiment is positive. No further action required.")
        # sentiment_json["customer_id"] = target_customer_id
        save_json_output("sentiment.json", sentiment_json, target_customer_id)
        return

    # Step 3: If NEUTRAL → GO DIRECTLY TO RETENTION STRATEGY
    elif sentiment_label == "neutral":
        # print("\nℹ️ Sentiment is neutral. Proceeding directly to Retention Strategy.")
        # sentiment_json["customer_id"] = target_customer_id
        save_json_output("sentiment.json", sentiment_json, target_customer_id)
        retention_inputs = {
            "call_transcript": call_transcript,
            "sentiment_agent_output": sentiment_json,
            "customer_profile": customer_profile,
            "root_cause_output": "",
            "churn_prediction_output": {}
        }
        retention_output = retention_crew.crew().kickoff(inputs=retention_inputs)
        retention_json = clean_and_convert_to_json(retention_output.raw)
        # print("\n🟢 Retention Strategy Output:\n", retention_json)
        # log_to_file("Retention Strategy Output", retention_json)
        # retention_json["customer_id"] = target_customer_id
        save_json_output("retention_recommender.json", retention_json, target_customer_id)

    # Step 4: If NEGATIVE → RCA → CHURN → (If needed) RETENTION
    elif sentiment_label == "negative":
        print("\n⚠️ Sentiment is negative. Starting Root Cause Analysis.")
        root_cause_inputs = {
            "call_transcript": call_transcript,
            "sentiment_agent_output": sentiment_json,
            "customer_profile": customer_profile
        }
        rca_output = root_cause_crew.crew().kickoff(inputs=root_cause_inputs)
        rca_json = clean_and_convert_to_json(rca_output.raw)
        # print("\n🔵 Root Cause Analysis Output:\n", rca_json)
        # log_to_file("Root Cause Analysis Output", rca_json)
        # rca_json["customer_id"] = target_customer_id
        save_json_output("rootcause.json", rca_json, target_customer_id)

        churn_inputs = {
            "call_transcript": call_transcript,
            "sentiment_agent_output": sentiment_json,
            "customer_profile": customer_profile,
            "root_cause_output": rca_json.get("root_cause", "")
        }
        churn_output = churn_crew.crew().kickoff(inputs=churn_inputs)
        churn_json = clean_and_convert_to_json(churn_output.raw)
        if not churn_json:
            print("❌ Churn prediction failed or returned invalid JSON.")
            log_to_file("Churn Prediction Output", "❌ Failed or invalid JSON.")
            return
        # print("\n🟠 Churn Prediction Output:\n", churn_json)
        # log_to_file("Churn Prediction Output", churn_json)
        # churn_json["customer_id"] = target_customer_id
        save_json_output("churn_risk.json", churn_json, target_customer_id)

        churn_label = churn_json.get("churn_risk_label", "").lower()
        if churn_label in ["high", "medium"]:
            print("\n⚠️ Churn risk is high or medium. Recommending retention strategy.")
            retention_inputs = {
                "call_transcript": call_transcript,
                "sentiment_agent_output": sentiment_json,
                "customer_profile": customer_profile,
                "root_cause_output": rca_json.get("root_cause", ""),
                "churn_prediction_output": churn_json
            }
            retention_output = retention_crew.crew().kickoff(inputs=retention_inputs)
            retention_json = clean_and_convert_to_json(retention_output.raw)
            # print("\n🟢 Retention Strategy Output:\n", retention_json)
            # log_to_file("Retention Strategy Output", retention_json)
            # retention_json["customer_id"] = target_customer_id
            save_json_output("retention_recommender.json", retention_json, target_customer_id)
        else:
            # print("\n✅ Churn risk is low. No retention strategy required.")
            # log_to_file("Pipeline Decision", "Churn risk is low. No retention strategy required.")
            # retention_json["customer_id"] = target_customer_id
            save_json_output("retention_recommender.json", retention_json, target_customer_id)

    else:
        print("\nUnknown sentiment value. Skipping.")
        # log_to_file("Pipeline Error", f"Unknown sentiment value: {sentiment_label}")











# if __name__ == "__main__":
#     clean_output_folder()
#     # File paths
#     CSV_CUSTOMER_PATH = "/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/Input/Customer_Data.csv"
#     CSV_CALL_PATH = "/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/Input/Call_Transcript_Data.csv"

#     # Choose a specific customer
#     target_customer_id = "CUST_110"

#     # Load data
#     df_customers = pd.read_csv(CSV_CUSTOMER_PATH)
#     df_calls = pd.read_csv(CSV_CALL_PATH)

#     # Get customer profile row
#     customer_row = df_customers[df_customers["customer_id"] == target_customer_id]
#     if customer_row.empty:
#         print(f"No customer found with ID: {target_customer_id}")
#         exit()
#     customer_data = customer_row.iloc[0].to_dict()

#     # Get call transcript row
#     call_row = df_calls[df_calls["customer_id"] == target_customer_id]
#     if call_row.empty:
#         print(f"No call transcript found for customer: {target_customer_id}")
#         exit()

#     # Optional: parse call_date
#     df_calls["call_date"] = pd.to_datetime(df_calls["call_date"], errors="coerce")
#     call_data = call_row.iloc[0]

#     # Run Customer Profile Summarizer
#     summarizer_crew = CustomerProfileSummarizerCrew()
#     customer_profile_output = summarizer_crew.crew().kickoff(inputs=customer_data)
#     customer_profile_json = clean_and_convert_to_json(customer_profile_output.raw)
#     if not customer_profile_json:
#         print("Failed to extract profile summary.")
#         exit()
#     customer_profile = customer_profile_json.get("summary", "")
#     customer_profile_json["customer_id"] = target_customer_id
#     # print("\nCustomer Profile Summary:\n", customer_profile)
#     save_json_output("customer_profile.json", customer_profile_json)

#     # Construct context phrase + transcript
#     call_transcript_summary = (
#         f"The following call took place on {call_data['call_date']} at {call_data['call_time']} "
#         f"between customer {call_data['customer_id']} and agent {call_data['agent_id']}. "
#         f"The call lasted for {call_data['call_duration']} minutes.\n\n"
#     )

#     # Full transcript with summary prefix
#     call_transcript = call_transcript_summary + call_data["transcript"]

#     # Create structured JSON explicitly
#     call_transcript_json = {
#         "call_id": call_data["call_id"],  
#         "call_date": str(call_data["call_date"].date()) if isinstance(call_data["call_date"], pd.Timestamp) else call_data["call_date"],
#         "call_time": call_data["call_time"],
#         "call_duration": call_data["call_duration"],
#         "agent_id": call_data["agent_id"],
#         "customer_id": target_customer_id,
#         "transcript": call_data["transcript"],
#         "context_summary": call_transcript_summary.strip()
#     }

#     # Save the JSON
#     save_json_output("call_transcript.json", call_transcript_json)

    
#     # call_transcript = call_transcript_summary + call_data["transcript"]

#     # call_transcript_json = call_data.to_dict()
#     # call_transcript_json["customer_id"] = target_customer_id
#     # call_transcript_json["context_summary"] = call_transcript_summary

#     #     # print("\nFinal Call Transcript Input:\n", call_transcript)
#     # save_json_output("call_transcript.json", call_transcript_json)


#     # Run the pipeline
#     print("\n" + "#"*80)
#     print("RUNNING PIPELINE")
#     print("#"*80)
#     pipeline(call_transcript, customer_profile)