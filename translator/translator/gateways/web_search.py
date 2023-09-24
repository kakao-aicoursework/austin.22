import os

from langchain.utilities import GoogleSearchAPIWrapper
from langchain.tools import Tool


class WebSearch:
    def __init__(self):
        self.search = GoogleSearchAPIWrapper(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID")
        )

        self.search_tool = Tool(
            name="Google Search",
            description="Search Google for recent results.",
            func=self.search.run,
        )

    def query_web_search(self, user_message: str) -> str:
        return self.search_tool.run(user_message)