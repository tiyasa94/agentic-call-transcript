from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
import litellm
from dotenv import load_dotenv
load_dotenv()

litellm._turn_on_debug()

# ✅ Configure OpenRouter LLM
llm = LLM(
    model="openrouter/meta-llama/llama-4-scout:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    litellm_provider="openrouter",
    max_tokens=1000
)


@CrewBase
class ChurnPredictionCrew:
    """Churn Risk Prediction Crew"""
    agents_config ="/Users/tiyasamukherjee/Desktop/Projects/agentic-call-transcript/src/config/agents.yaml"
    tasks_config = "/Users/tiyasamukherjee/Desktop/Projects/agentic-call-transcript/src/config/tasks_churn_risk.yaml"

    @agent
    def churn_prediction_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["churn_prediction_agent"],
            llm=llm
        )

    @task
    def churn_prediction_task(self) -> Task:
        return Task(
            config=self.tasks_config["churn_prediction_task"]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


