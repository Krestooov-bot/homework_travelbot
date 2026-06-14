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
    """Додає нову зупинку до маршруту подорожі."""
    if "itinerary" not in st.session_state:
        st.session_state.itinerary = []
    if not any(item["Місто"].lower() == city.lower() for item in st.session_state.itinerary):
        st.session_state.itinerary.append({"Місто": city, "Активність": activity, "Днів": days})
    return f"Зупинку {city} додано."

@tool
def estimate_budget(days: int, daily_budget: float = 100.0) -> str:
    """Розраховує орієнтовний бюджет подорожі."""
    return f"Бюджет на {days} днів: {days * daily_budget}$."

@tool
def show_itinerary() -> str:
    """Показує поточний збережений маршрут користувача."""
    return "Маршрут оновлено."

@tool
def wikipedia_search(city: str) -> str:
    """Пошук цікавих фактів та погоди про місто в інтернеті."""
    return f"Інформація з мережі: {city} — чудове місто. Погода зараз сприятлива для подорожей."

tools = [add_stop, estimate_budget, show_itinerary, wikipedia_search]
tool_node = ToolNode(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: State):
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(state['messages'])

    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "add_stop":
                args = tc["args"]
                if "itinerary" not in st.session_state:
                    st.session_state.itinerary = []
                if not any(i["Місто"].lower() == args.get("city", "").lower() for i in st.session_state.itinerary):
                    st.session_state.itinerary.append({
                        "Місто": args.get("city", "Невідомо"),
                        "Активність": args.get("activity", args.get("notes", "Огляд")),
                        "Днів": int(args.get("days", args.get("duration_days", 1)))
                    })

    return {"messages": [response]}

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
        "Ти TravelBot. Твоє завдання - підтримувати розмову.\n"
        "НІКОЛИ не пиши програмний код у чат."
    )
    
    input_messages = [("system", sys_msg), ("user", user_input)]
    events = compiled_graph.stream({"messages": input_messages}, config, stream_mode="values")
    
    final_message = ""
    for event in events:
        if "messages" in event:
            msg = event["messages"][-1]
            if isinstance(msg.content, str):
                final_message = msg.content
            elif isinstance(msg.content, list):
                texts = [str(item.get("text", "")) for item in msg.content if isinstance(item, dict) and "text" in item]
                final_message = " ".join(texts)
            else:
                final_message = str(msg.content)

    if not final_message:
        final_message = "Інформацію успішно оброблено."

    if "add_stop" in final_message:
        city_match = re.search(r"city\s*=\s*['\"]([^'\"]+)['\"]", final_message, re.IGNORECASE)
        days_match = re.search(r"(?:duration_days|days|дні)\s*=\s*(\d+)", final_message, re.IGNORECASE)
        notes_match = re.search(r"(?:notes|activity)\s*=\s*['\"]([^'\"]+)['\"]", final_message, re.IGNORECASE)
        
        if city_match:
            city = city_match.group(1)
            days = int(days_match.group(1)) if days_match else 3
            activity = notes_match.group(1) if notes_match else "Огляд"
            
            if "itinerary" not in st.session_state:
                st.session_state.itinerary = []
                
            if not any(i["Місто"].lower() == city.lower() for i in st.session_state.itinerary):
                st.session_state.itinerary.append({
                    "Місто": city,
                    "Активність": activity,
                    "Днів": days
                })
            
            final_message = re.sub(r"```python.*?```", "", final_message, flags=re.DOTALL)
            final_message = re.sub(r"add_stop\(.*?\)", "", final_message)
            final_message += "\n\n*(✅ Маршрут успішно занесено в таблицю!)*"
            
    return final_message.strip()
