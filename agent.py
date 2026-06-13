import os
import re
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

@tool
def add_stop(city: str, duration_days: int, notes: str) -> str:
    """Always use this tool to add a destination to the travel itinerary."""
    if "itinerary" not in st.session_state:
        st.session_state.itinerary = []
    
    st.session_state.itinerary.append({
        "Місто": city,
        "Активність": notes,
        "Днів": duration_days
    })
    return f"Successfully added {city} for {duration_days} days."

@tool
def estimate_budget(days: int, daily_budget: float = 100.0) -> str:
    """Use this tool to calculate the estimated budget."""
    total = days * daily_budget
    return f"Estimated budget for {days} days is {total}$."

tools = [add_stop, estimate_budget]

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)

def run_agent_chat(user_input: str, thread_id: str, system_prompt: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    messages = [
        ("system", "You are a helpful travel planner. You MUST use the provided tools to save data. DO NOT output raw python code like `tool_code:add_stop()`."),
        ("user", user_input)
    ]
    
    response = agent_executor.invoke({"messages": messages}, config)
    final_message = response["messages"][-1].content
    

    if "add_stop" in final_message and "city=" in final_message:
        try:
            city_match = re.search(r"city=['\"]([^'\"]+)['\"]", final_message)
            days_match = re.search(r"duration_days=(\d+)", final_message)
            notes_match = re.search(r"notes=['\"]([^'\"]+)['\"]", final_message)
            
            if city_match and days_match:
                if "itinerary" not in st.session_state:
                    st.session_state.itinerary = []
                
                st.session_state.itinerary.append({
                    "Місто": city_match.group(1),
                    "Активність": notes_match.group(1) if notes_match else "Не вказано",
                    "Днів": int(days_match.group(1))
                })

                final_message += "\n\n*(✅ Системне перехоплення: Маршрут успішно занесено в таблицю!)*"
        except Exception as e:
            pass
            
    return final_message
