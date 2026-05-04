from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-course-py!")
    information = """
Prakhar
    """
    """// this information we are going to propagate to LLM"""

    summary_template = """
    Given the name of a peron {information}, I want you to greet the person, with his name.
    """
    summary_prompt_template = PromptTemplate(
        input_variables = ["information"], template = summary_template
    )
    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # an object which will interact with model
    # i had to use a basic model "gemini-2.5-flash" to even generate an ouput
    llm = ChatOllama(temperature = 0.4, model = "qwen2.5")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)



if __name__ == "__main__":
    main()
