import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

@tool
def add_stop(city: str, activity: str, days: int):
    """Додає нову зупинку до маршруту подорожі."""
    if "itinerary" not in st.session_state:
        st.session_state.itinerary = []
    
    st.session_state.itinerary.append({
        "Місто": city,
        "Активність": activity,
        "Днів": days
    })
    return f"Зупинку в місті {city} успішно додано до плану."

@tool
def estimate_budget(days: int, daily_budget: float = 100.0):
    """Розраховує орієнтовний бюджет подорожі."""
    total = days * daily_budget
    return f"Орієнтовний бюджет на {days} днів становить {total}$."

@tool
def show_itinerary():
    """Показує поточний збережений маршрут користувача."""
    itinerary = st.session_state.get("itinerary", [])
    if not itinerary:
        return "Маршрут поки що порожній."
    
    res = "Поточний маршрут:\n"
    for item in itinerary:
        res += f"- {item['Місто']} ({item['Днів']} днів): {item['Активність']}\n"
    return res

tools = [add_stop, estimate_budget, show_itinerary]
tool_node = ToolNode(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: State):
    # ВАЖЛИВО: Використовуємо стабільну 1.5-flash для роботи з інструментами
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.2)
    model_with_tools = model.bind_tools(tools)
    return {"messages": [model_with_tools.invoke(state['messages'])]}

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

memory = MemorySaver()
compiled_graph = workflow.compile(checkpointer=memory)

def run_agent_chat(user_input: str, thread_id: str, system_prompt: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    input_messages = []
    if system_prompt:
        input_messages.append(("system", system_prompt))
    input_messages.append(("user", user_input))
    
    events = compiled_graph.stream({"messages": input_messages}, config, stream_mode="values")
    
    final_message = ""
    for event in events:
        if "messages" in event:
            msg = event["messages"][-1]
            if msg.content:
                final_message = msg.content
                
    return final_message
