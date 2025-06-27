from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
import litellm
litellm._turn_on_debug()

llm = LLM(
    api_key=os.getenv("WATSONX_APIKEY"),
    model="watsonx/meta-llama/llama-3-3-70b-instruct",
    # base_url="https://api.watsonx.ai/v1",
	
	base_url=os.getenv("WATSONX_URL"),
    max_tokens=1000
)


@CrewBase
class ChurnPredictionCrew():
	"""Sentiment Tone Analyzer Crew"""

	agents_config = '/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/src/config/agents.yaml'
	tasks_config = '/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/src/config/tasks_churn_risk.yaml'

	@agent
	def churn_prediction_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['churn_prediction_agent'],
			llm=llm
		)

	@task
	def churn_prediction_task(self) -> Task:
		return Task(
			config=self.tasks_config['churn_prediction_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Churn Prediction Crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)


