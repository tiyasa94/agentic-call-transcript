import os
import pandas as pd
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import litellm
from dotenv import load_dotenv
load_dotenv()

# Enable debug mode
litellm._turn_on_debug()

# Initialize LLM
# ✅ Configure OpenRouter LLM
llm = LLM(
    model="openrouter/meta-llama/llama-4-scout:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    litellm_provider="openrouter",
    max_tokens=1000
)



# Path to input data
CSV_PATH = '/Users/tiyasamukherjee/Desktop/Projects/agentic-call-transcript/Input/Customer_Data.csv'

@CrewBase
class CustomerProfileSummarizerCrew():
    """Customer Profile Summarizer Crew"""

    agents_config = '/Users/tiyasamukherjee/Desktop/Projects/agentic-call-transcript/src/config/agents.yaml'
    tasks_config = '/Users/tiyasamukherjee/Desktop/Projects/agentic-call-transcript/src/config/tasks_customer_profile.yaml'

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




