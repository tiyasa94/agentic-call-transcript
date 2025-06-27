import os
import pandas as pd
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import litellm

# Enable debug mode
litellm._turn_on_debug()

# Initialize LLM
llm = LLM(
    api_key=os.getenv("WATSONX_APIKEY"),
    model="watsonx/meta-llama/llama-3-3-70b-instruct",
    base_url=os.getenv("WATSONX_URL"),
    max_tokens=1000
)

# Path to input data
CSV_PATH = '/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/Input/Customer_Data.csv'

@CrewBase
class CustomerProfileSummarizerCrew():
    """Customer Profile Summarizer Crew"""

    agents_config = '/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/src/config/agents.yaml'
    tasks_config = '/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/src/config/tasks_customer_profile.yaml'

    @agent
    def customer_profile_summarizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['customer_profile_summarizer_agent'],
            llm=llm
        )

    @task
    def customer_profile_summarization_task(self) -> Task:
        return Task(
            config=self.tasks_config['customer_profile_summarization_task']
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Customer Profile Summarizer Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

def run_customer_profile_summaries():
    # Load CSV data
    df = pd.read_csv(CSV_PATH)

    # Initialize the Crew
    summarizer = CustomerProfileSummarizerCrew()
    crew = summarizer.crew()

    for _, row in df.iterrows():
        input_data = {
            "customer_id": row["customer_id"],
            "tenure_months": row["tenure_months"],
            "contract_type": row["contract_type"],
            "plan_type": row["plan_type"],
            "payment_method": row["payment_method"],
            "avg_monthly_spend": row["avg_monthly_spend"],
            "total_interactions": row["total_interactions"],
            "unresolved_issues": row["unresolved_issues"],
            "recent_outage_count": row["recent_outage_count"],
            "loyalty_tier": row["loyalty_tier"]
        }

        print(f"\n🔍 Generating profile for Customer ID: {row['customer_id']}")
        result = crew.run(inputs=input_data)
        print(f"📝 Summary: {result['customer_profile_summarization_task']['output']}")




