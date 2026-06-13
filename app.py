import streamlit as st
import pandas as pd
import json
import uuid
import os
from google import genai
from google.genai import types
from agent import run_agent_chat

st.set_page_config(page_title="TravelBot AI", page_icon="✈️", layout="wide")

st.markdown("""
<style>
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    .stMetric { background-color: rgba(128, 128, 128, 0.1); padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

api_key = st.secrets.get("GOOGLE_API_KEY")

@st.cache_resource
def get_client():
    if not api_key:
        st.error("Додайте GOOGLE_API_KEY у .streamlit/secrets.toml")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_client()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "itinerary" not in st.session_state:
    st.session_state.itinerary = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "token_count" not in st.session_state:
    st.session_state.token_count = 0

with st.sidebar:
    st.title("⚙️ Налаштування")
    
    chat_mode = st.radio("Режим роботи:", ["Звичайний чат (Стрімінг)", "Агент з інструментами"])
    
    st.divider()
    temperature = st.slider("Температура (Креативність)", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.number_input("Max Tokens", 100, 8192, 1500)
    
    system_prompt = st.text_area(
        "Системний промпт:", 
        "Ти TravelBot. Допомагай планувати подорожі. Якщо користувач просить додати місто, обов'язково використовуй інструмент add_stop."
    )
    
    st.divider()
    if st.button("🗑️ Очистити історію"):
        st.session_state.messages = []
        st.session_state.itinerary = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.token_count = 0
        st.rerun()

st.title("✈️ AI Travel Planner")

tab1, tab2 = st.tabs(["💬 Планування (Чат)", "🗺️ Ваш маршрут та Експорт"])

with tab1:
    chat_col, stats_col = st.columns([3, 1])
    
    with stats_col:
        st.metric("Повідомлень", len(st.session_state.messages))
        st.metric("Токенів (оцінка)", st.session_state.token_count)
    
    with chat_col:
        if not st.session_state.messages:
            st.chat_message("assistant").write("Привіт! Я TravelBot. Скажи куди ти хочеш поїхати або попроси мене додати місто до твого маршруту!")

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Наприклад: Склади план в Париж на 3 дні і додай його в таблицю"):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Складаю маршрут..."):
                    
                    if chat_mode == "Звичайний чат (Стрімінг)":
                        contents = []
                        config = types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=temperature,
                            max_output_tokens=max_tokens
                        )
                        for m in st.session_state.messages:
                            role = "model" if m["role"] == "assistant" else "user"
                            safe_text = str(m.get("content", ""))
                            if not safe_text.strip():
                                safe_text = " "
                            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=safe_text)]))
                        
                        try:
                            def stream():
                                response = client.models.generate_content_stream(
                                    model='gemini-2.5-flash', contents=contents, config=config
                                )
                                for chunk in response: yield chunk.text
                            full_response = st.write_stream(stream())
                        except Exception as e:
                            full_response = f"Помилка API: {e}"
                            st.error(full_response)
                            
                    else:
                        try:
                            full_response = run_agent_chat(prompt, st.session_state.thread_id, system_prompt)
                            st.write(full_response)
                        except Exception as e:
                            full_response = f"Помилка Агента: {e}"
                            st.error(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.token_count += int((len(prompt.split()) + len(full_response.split())) * 1.3)
            st.rerun()

with tab2:
    st.subheader("Поточний план подорожі")
    
    if st.session_state.itinerary:
        df = pd.DataFrame(st.session_state.itinerary)
        
        m1, m2 = st.columns(2)
        m1.metric("Кількість міст", len(df))
        m2.metric("Загальна кількість днів", df["Днів"].sum())
        
        st.dataframe(df, use_container_width=True)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Завантажити маршрут (CSV)", data=csv, file_name="itinerary.csv", mime="text/csv")
        with col_dl2:
            chat_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
            st.download_button("📥 Завантажити історію чату (JSON)", data=chat_json, file_name="chat.json", mime="application/json")
    else:
        st.info("Маршрут порожній. Попросіть агента додати зупинки або використайте форму нижче.")

    with st.expander("➕ Додати зупинку вручну (без ШІ)"):
        with st.form("manual_add"):
            c_name = st.text_input("Місто")
            c_act = st.text_input("Що робити?")
            c_days = st.number_input("Кількість днів", 1, 30, 1)
            if st.form_submit_button("Зберегти"):
                if c_name:
                    st.session_state.itinerary.append({"Місто": c_name, "Активність": c_act, "Днів": c_days})
                    st.success("Додано!")
                    st.rerun()
