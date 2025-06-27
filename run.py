from src.agents.churn_prediction import ChurnPredictionCrew

if __name__ == "__main__":
    crew_instance = ChurnPredictionCrew()
    result = crew_instance.crew().run()
    print("\n🔍 Final Output:\n", result)
