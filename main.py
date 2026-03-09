from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    print("Hello from langchain-course-py!")
    information = """
Avul Pakir Jainulabdeen Abdul Kalam ( born on 15 October 1931 – 27 July 2015) was an Indian aerospace scientist and statesman who served as the president of India from 2002 to 2007.
Born and raised in a Muslim family in Rameswaram, Tamil Nadu, Kalam studied physics and aerospace engineering. He was known as the "Missile Man of India" for his work on the development of ballistic missile and launch vehicle technology. 
Kalam was elected as the president of India in 2002.
    """
    """// this information we are going to propoage to LLM"""

    summary_template = """
    Given the information {information} about a person I want you to create:
    1. a short summary of the information
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables = ["information"], template = summary_template
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # an object which will interact with model
    # i had to use a basic model "gemini-2.5-flash" to even generate an ouput
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)



if __name__ == "__main__":
    main()
