from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage # this gives an interface for all models, switching models is very easy

from langsmith import traceable

# tool message is going to be containing tool result

MAX_ITERATIONS = 10 # limit the runs
MODEL = "qwen3:1.7b"

# --- Tools (Langchain @tool decorator) ---

def getCatalog():
    return {"laptop": 1299.99, "headphones": 14.95, "keyboard": 89.55}

def get_discount_percentages():
    return {"bronze": 5, "silver": 12, "gold": 23}
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"   >>> Executing get_prouct_price(product='{product}')")
    prices = getCatalog()
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price, round to 2 decimal places.
    Available tiers: bronze, silver, gold."""
    print(f"   >>> Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    discount_percentages = get_discount_percentages()
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)

# --- Agent loop ---

@traceable(name="Langchain Agent Loop") #name will help is trace everything inside the function under langchain
def run_agent(question: str):
    tools = [get_product_price, apply_discount] 
    tools_dict = {t.name: t for t in tools} # this is goin to help us to take the result of LLM (tool name) and get python object (tool) we can execute, hence using it going to be much clear
    llm = init_chat_model(f"ollama: {MODEL}", temperature=0) # this is more convient here we dont need to import the object of chat model itself
    # now let our model know the known tools
    llm_with_tools = llm.bind_tools(tools) # this .bind_tools is avaliable for all LLM supporting funtion calling capability.
    print(f"Question: {question}")
    print("="* 60)

    # now need to create brain of the LLM(basically bunch of prompts sending to LLms, which will leverage its reasoning capabilities and use as reasoning agent to decide what's the answer whether need to execute tool)
    #System messages represent system prompts
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. pass the exact price "
                "return by get_product_price - do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. if the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            )
        ),
        HumanMessage(content= question), 
    ]

    for it in range(1, 1 + MAX_ITERATIONS ):
        print(f"\n--- Iteration {it} ---")
        ai_msg = llm_with_tools.invoke(messages)
        # this is going to be a tool called decision of LLM or content in case it has the answer(when no tool call is needed)
        tool_calls = ai_msg.tool_calls
        if not tool_calls: 
            print(f"\nFinalAnswer: {ai_msg.content}")
            return ai_msg.content
        # Process only the FIRST tool call force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"   [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name) # this is goin to be the python funtion which we can invoke.

        if tool_to_use is None: 
            raise ValueError(f"Tool '{tool_name}' not found")
        
        # tools are runnable hence we can invoke them
        observation = tool_to_use.invoke(tool_args)

        print(f"   [Tool Result]: {observation}")
        # now every time we iterate, we want to append the result of the tool & to append the reasoning of the agent, it would have the history so it would know what to do.
        messages.append(ai_msg) # ai_msg contains the tool call
        # also append tool result
        messages.append(
            ToolMessage(content = str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: max iterations reached without a final answer")
    return None





if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools) !")
    print()
    result = run_agent("what is the price of a 'laptop' after applying a gold discount?")