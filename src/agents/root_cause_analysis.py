from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
import litellm
litellm._turn_on_debug()
from dotenv import load_dotenv
load_dotenv()

# ✅ Configure OpenRouter LLM
llm = LLM(
    model="openrouter/meta-llama/llama-4-scout:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    litellm_provider="openrouter",
    max_tokens=1000
)



@CrewBase
class RootCauseAnalysisCrew():
	"""Root Cause Analysis Crew"""

	agents_config = '/Users/tiyasamukherjee/Desktop/Projects/agentic-call-transcript/src/config/agents.yaml'
	tasks_config = '/Users/tiyasamukherjee/Desktop/Projects/agentic-call-transcript/src/config/tasks_rootcause.yaml'

	@agent
	def root_cause_analysis_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['root_cause_analysis_agent'],
			llm=llm
		)

	@task
	def root_cause_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['root_cause_analysis_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Root Cause Analysis Crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)







