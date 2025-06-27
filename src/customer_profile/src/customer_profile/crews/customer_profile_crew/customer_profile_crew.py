from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class CustomerProfileCrew():
	"""Poem Crew"""

	agents_config = '/Users/munendrasingh/Desktop/INDUSTRY_ASSETS/call_transcript_analytics/src/customer_profile/src/customer_profile/crews/poem_crew/config/agents.yaml'
	tasks_config = '/Users/munendrasingh/Desktop/INDUSTRY_ASSETS/call_transcript_analytics/src/customer_profile/src/customer_profile/crews/poem_crew/config/tasks.yaml'

	@agent
	def profile_generator(self) -> Agent:
		return Agent(
			config=self.agents_config['profile_generator'],
		)
	
	@agent
	def profile_evaluator(self) -> Agent:
		return Agent(
			config=self.agents_config['profile_evaluator'],
		)

	@task
	def profile_task(self) -> Task:
		return Task(
			config=self.tasks_config['profile_task'],
		)
	
	@task
	def evaluation_task(self) -> Task:
		return Task(
			config=self.tasks_config['evaluation_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Research Crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)