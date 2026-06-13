import os
import re
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
def add_stop(city: str, activity: str, days: int) -> str:
    """Додає нову зупинку до маршруту подорожі. ОБОВ'ЯЗКОВО використовуй цей інструмент."""
    if "itinerary" not in st.session_state:
        st.session_state.itinerary = []
        
    if not any(item["Місто"].lower() == city.lower() for item in st.session_state.itinerary):
        st.session_state.itinerary.append({
            "Місто": city,
            "Активність": activity,
            "Днів": days
        })
        return f"Успіх! Зупинку {city} збережено в таблиці."
    return f"Зупинка {city} вже є в таблиці."

@tool
def estimate_budget(days: int, daily_budget: float = 100.0) -> str:
    """Розраховує орієнтовний бюджет подорожі."""
    return f"Бюджет на {days} днів: {days * daily_budget}$."

@tool
def show_itinerary() -> str:
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
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.1)
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
    
    sys_msg = (
        "Ти TravelBot. Твоє завдання - керувати маршрутом.\n"
        "Якщо користувач просить додати місто, ОБОВ'ЯЗКОВО використовуй інструмент add_stop.\n"
        "НІКОЛИ не пиши виклик функції як текст."
    )
    
    input_messages = [("system", sys_msg), ("user", user_input)]
    events = compiled_graph.stream({"messages": input_messages}, config, stream_mode="values")
    
    final_message = ""
    for event in events:
        if "messages" in event:
            msg = event["messages"][-1]
            if msg.content:
                final_message = msg.content
                
    city_match = re.search(r"(?:city|місто)\s*=\s*['\"]([^'\"]+)['\"]", final_message, re.IGNORECASE)
    days_match = re.search(r"(?:duration_days|days|дні)\s*=\s*(\d+)", final_message, re.IGNORECASE)
    
    if city_match:
        city = city_match.group(1)
        days = int(days_match.group(1)) if days_match else 3
        
        if "itinerary" not in st.session_state:
            st.session_state.itinerary = []
            
        if not any(i["Місто"].lower() == city.lower() for i in st.session_state.itinerary):
            st.session_state.itinerary.append({
                "Місто": city,
                "Активність": "Заплановано ШІ",
                "Днів": days
            })
        final_message = final_message.replace(city_match.group(0), "")
            
    return final_message
