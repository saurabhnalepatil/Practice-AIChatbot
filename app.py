from datetime import datetime
import json
import operator
import os
from typing import Annotated, Sequence, TypedDict

import dotenv
import pytz
import requests
import streamlit as st
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation


dotenv.load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = "RetailBuddy API"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


@tool
def call_external_api(search_term: str) -> str:
    """Fetches resources data from the Catalyst API, categorized by GOODS, FOOD, HOUSING, TRANSIT, HEALTH, MONEY, WORK, or LEGAL.

    Args:
        search_term (str): The user's search query, specifying the desired resource category (e.g., "food", "housing", "legal").

    Returns:
        A dictionary containing the API response data or an error message on failure.
    """
    subcat_id = 0
    cat_id = 0
    maintopic_id = 0
    user_zip = 0
    url = f"http://catalystws.celluphone.com/api/ResourcesLandings/{search_term}/{subcat_id}/{cat_id}/{maintopic_id}/{user_zip}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return str(data[:1])
    else:
        return None


@tool
def get_outlet_products() -> list:
    """
    Fetches and processes product details for a specific company and outlet.

    Parameters:
    - company_id (int): The ID of the company.
    - outlet_id (int): The ID of the outlet.

    Returns:
    list: A list of dictionaries containing filtered product details or an error message.
    """
    try:
        company_id = 1
        outlet_id = 1
        base_url = "https://rcsdev.azurewebsites.net/POSProducts/getOutletProducts"
        params = {"companyId": company_id, "outletId": outlet_id}

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        products = data.get("product", [])

        desired_fields = [
            "productID", "productName", "productDescr", "productTypeID", "isTrackInventory",
            "inventoryTrackingID", "unitID", "isReturnable", "createdUserID", "createdDt",
            "active", "imageIcon", "categoryTypeID", "categoryID", "brandID", "count",
            "inventory", "supplierID", "companyID", "costPrice", "sellingPrice",
            "availableStock", "stockReorderPoint", "outletsIds", "discount", "inventoryID"
        ]

        filtered_products = [
            {key: product.get(key) for key in desired_fields} for product in products
        ]

        return filtered_products

    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching product details: {e}"
        return [{"error": "Failed to fetch product details", "details": error_message}]


@tool
def get_sales_summary() -> list:
    """
    Fetches and processes sales summary details for a specific company for the last two days. 

    This function is used to retrieve sales data that includes yesterday's sales details and compares them with today's data. 
    It is particularly designed for AI chatbot integration to respond to user queries such as "What was yesterday's total sale in all outlets?"

    Parameters:
    None

    Returns:
    list: A list of dictionaries containing sales summary details for the last two days, 
          including key metrics such as revenue, gross profit, sales count, and more. 
          If an error occurs, a dictionary with an error message and details is returned.
    """
    try:
        india_timezone = pytz.timezone("Asia/Kolkata")
        today_date = datetime.now(india_timezone).strftime("%Y-%m-%dT00:00:00.000Z")
        base_url = "https://rcsdev.azurewebsites.net/POSReports/getSellSemmeryDetails"
        params = {
            "numPeriods": 2,
            "periodType": "day",
            "reportType": "ByOutlet",
            "startDate": today_date,
            # "startDate": "2024-11-16T00:00:00.000Z",
            "companyId": 1
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        reports = data.get("reportTypeTbl", [])

        desired_fields = [
            "id", "name", "revenue", "revenueWithTax", "costOfGoodsSold", "grossProfit", "margin",
            "marginAmount", "firstSale", "lastSale", "itemSold", "avgSalesValue", "avgSalesWithTaxValue",
            "salesCount", "customerCount", "averageItemSale", "discountPercentage", "discount", "tax",
            "date", "returnCount", "returnPercentage", "returnAmount", "saleWithCustemoraAttached", "rev"
        ]

        filtered_reports = [
            {key: report.get(key) for key in desired_fields} for report in reports
        ]

        return filtered_reports

    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching sales summary details: {e}"
        return [{"error": "Failed to fetch sales summary details", "details": error_message}]


@tool
def get_outlet_details() -> list:
    """
    Fetches and processes outlet details for a specific user.

    Parameters:
    user_id (int): The ID of the user to fetch outlet details for.

    Returns:
    list: A list of dictionaries containing filtered outlet details or an error message.
    """
    try:
        user_id = 1166
        base_url = "https://rcsdev.azurewebsites.net/PosOutlets/getUserOutlets"
        params = {
            "userId": user_id
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        outlets = response.json()

        desired_fields = [
            "outletID", "outletName", "location", "timeZoneID", "taxID", "taxName",
            "taxRate", "createdDt", "lastUpdatedDt", "createdByUserID", "lastUpdatedByUserID",
            "companyID", "userID", "isActive", "isSelected", "isPreferenceSelected",
            "counters", "previousCounters"
        ]

        filtered_outlets = [
            {key: outlet.get(key) for key in desired_fields} for outlet in outlets
        ]

        return filtered_outlets

    except requests.RequestException as e:
        return [{"error": str(e)}]


tools = [get_outlet_products, get_sales_summary, get_outlet_details]

tool_executor = ToolExecutor(tools)


def should_retrieve(state: AgentState) -> str:
    """Decides whether the agent should retrieve more information or end the process.

    Args:
        state (AgentState): The current state

    Returns:
        str: A decision to either "continue" the retrieval process or "end" it
    """
    messages = state["messages"]
    last_message = messages[-1]

    if "function_call" not in last_message.additional_kwargs:
        return "end"
    else:
        return "continue"


def agent(state: AgentState) -> AgentState:
    """Invokes the agent model to generate a response based on the current state.

    Args:
        state (AgentState): The current state

    Returns:
        AgentState: The updated state with the agent response appended to messages
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant named RetailBuddy who is very interative and provides suggestions whenever required."),
            ("human", "{input}"),
        ]
    )
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True,
                       model="gpt-4o-mini")
    functions = [format_tool_to_openai_function(t) for t in tools]
    model = model.bind_functions(functions)
    chain = prompt | model
    response = chain.invoke(messages)
    return {"messages": [response]}


def retrieve(state: AgentState) -> AgentState:
    """Uses tool to execute retrieval.

    Args:
        state (AgentState): The current state

    Returns:
        AgentState: The updated state with retrieved docs
    """
    messages = state["messages"]
    last_message = messages[-1]
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]),
    )
    response = tool_executor.invoke(action)
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}


def initialize_prompt_template() -> PromptTemplate:
    """Initialize the prompt template for generating responses."""
    template = PromptTemplate.from_template(
        """You are a helpful assistant named RetailBuddy who is very interative and provides suggestions whenever required.
        When picking the search term from user question keep it very minimal, 
        for example if user wants to find child care service then the search term should be child care.
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know.
        Always recheck the user query and use the appropriate tools to find answers. 
        Also try to validate the Question and Response are relevant to each other or not.


        Question: {question} 

        Context: {context} 

        Answer:"""
    )
    return template


def generate(state: AgentState) -> AgentState:
    """Generate answer

    Args:
        state (AgentState): The current state

    Returns:
        AgentState: The updated state with re-phrased question
    """
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    prompt = initialize_prompt_template()
    llm = ChatOpenAI(model_name="gpt-4o-mini",
                     temperature=0, streaming=True)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("generate", generate)  # retrieval

# Call agent node to decide to retrieve or not
workflow.set_entry_point("agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    should_retrieve,
    {
        "continue": "retrieve",
        "end": END,
    },
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()


def main():
    st.title("RetailBuddy  - Your AI Assistant")

    user_input = st.text_input("Enter your question or query:", key="user_input")
    button_clicked = st.button("Ask RetailBuddy ")

    if button_clicked:
        state = {"messages": [HumanMessage(content=user_input)]}
        final_output = None
        for output in app.stream(state):
            final_output = output

        if "generate" in final_output:
            message = final_output["generate"]["messages"][0]
        elif "agent" in final_output:
            message = final_output["agent"]["messages"][0].content
        else:
            message = "Sorry, I did not find any information regarding this query."

        st.write(message)


if __name__ == "__main__":
    main()
