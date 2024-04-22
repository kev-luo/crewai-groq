import os
from crewai import Crew

from textwrap import dedent
from agents import CustomAgents
from tasks import CustomTasks

from dotenv import load_dotenv
load_dotenv()

class CustomCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.cities = cities
        self.origin = origin
        self.interests = interests
        self.date_range = date_range

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = CustomAgents()
        tasks = CustomTasks()

        # Custom agents
        city_selection_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()

        # Custom tasks
        identify_task = tasks.identify_task(
            city_selection_agent,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )

        gather_task = tasks.gather_task(
            local_expert_agent,
            self.origin,
            self.interests,
            self.date_range
        )

        plan_task = tasks.plan_task(
            travel_concierge_agent, 
            self.origin,
            self.interests,
            self.date_range
        )

        # Define your custom crew here
        crew = Crew(
            agents=[city_selection_agent, local_expert_agent, travel_concierge_agent],
            tasks=[identify_task, gather_task, plan_task],
            verbose=True,
        )

        result = crew.kickoff()
        return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":
    print("## Welcome to Trip Planner Crew")
    print('-------------------------------')
    location = input(
        dedent("""
        From where will you be traveling from?
        """))
    cities = input(
        dedent("""
        What are the cities options you are interested in visiting?
        """))
    date_range = input(
        dedent("""
        What is the date range you are interested in traveling?
        """))
    interests = input(
        dedent("""
        What are some of your high level interests and hobbies?
        """))
    
    trip_crew = CustomCrew(location, cities, date_range, interests)
    result = trip_crew.run()
    print("\n\n########################")
    print("## Here is you Trip Plan")
    print("########################\n")
    print(result)