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
class SentimentToneAnalyzerCrew():
	"""Sentiment Tone Analyzer Crew"""

	agents_config = '/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/src/config/agents.yaml'
	tasks_config = '/Users/tiyasamukherjee/Documents/GitHub/call_transcript_analytics/src/config/tasks_sentiment.yaml'

	@agent
	def transcript_sentiment_tone_analyzer_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['transcript_sentiment_tone_analyzer_agent'],
			llm=llm
		)

	@task
	def sentiment_analysis_task(self) -> Task:
		return Task(
			config=self.tasks_config['sentiment_analysis_task'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Sentiment and Tone Analyzer Crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)