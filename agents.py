from crewai import Agent
from textwrap import dedent
from langchain_groq import ChatGroq

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools


# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py
class CustomAgents:
    def __init__(self):
        self.GroqMixtral = ChatGroq(model="mixtral-8x7b-32768", temperature=0.7)
        self.GroqLlama3 = ChatGroq(model="llama3-70b-8192", temperature=0.7)

    def city_selection_agent(self):
        return Agent(
            role="City Selection Expert",
            goal=dedent(f"""Select the best city based on weather, season, and prices"""),
            backstory=dedent(f"""An expert in analyzing travel data to pick ideal destinations"""),
            tools=[
                SearchTools.search_internet,
                BrowserTools.scrape_and_summarize_website,
            ],
            allow_delegation=False,
            max_iter=2,
            memory=True,
            verbose=True,
            llm=self.GroqLlama3,
        )

    def local_expert(self):
        return Agent(
            role="Local Expert at this city",
            goal=dedent(f"""Provide the BEST insights about the selected city"""),
            backstory=dedent(f"""A knowledgeable local guide with extensive information about the city, it's attractions and customs"""),
            tools=[
                SearchTools.search_internet,
                BrowserTools.scrape_and_summarize_website,
            ],
            allow_delegation=False,
            max_iter=2,
            memory=True,
            verbose=True,
            llm=self.GroqLlama3,
        )

    def travel_concierge(self):
        return Agent(
            role="Amazing Travel Concierge",
            goal=dedent(f"""Create the most amazing travel itineraries with budget and packing suggestions for the city"""),
            backstory=dedent(f"""Specialist in travel planning and logistics with decades of experience"""),
            tools=[
                SearchTools.search_internet,
                BrowserTools.scrape_and_summarize_website,
                CalculatorTools.calculate,
            ],
            allow_delegation=False,
            max_iter=2,
            memory=True,
            verbose=True,
            llm=self.GroqLlama3,
        )