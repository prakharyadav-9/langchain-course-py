from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient() # here it will look for env variable TAVILY_API_KEY
tavily_search = TavilySearch() # this is a wrapper that tavily team has given for search check https://docs.tavily.com/welcome for more

# this is a custom search implemented, but tavily team has already has an implementation so better to use them
@tool
def search_internal(query: str) -> str: 
    """
    A Tool that searches about the query, when user askes about the `py-learning course` 

    Args:
        query: the query to search from the internet as a string
    Returns:
        the search result
    """
    print(f"we will search using tavily with query: {query}")
    # return tavily.search(query=query)
    return "py learning is going awesome"

@tool
def internal_tool_to_get_current_date() -> str:
    """
    A Tool that return the current local date YYYMMDD as string
    """
    return datetime().now().strftime("%Y%m%d")

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
tools = [search_internal
         , internal_tool_to_get_current_date
         , tavily_search
         ]
agent = create_agent(model = llm, tools= tools)

def main():
    print("Hello from langchain-course-py!")
    print("Hey!! How can i help you today?")
    content = input("type your query: ")
    # agent runnable 
    result = agent.invoke({"messages":[HumanMessage(content=content)]})
    print("Yup, the agent returned and answer...")
    print(result)


if __name__ == "__main__":
    main()
