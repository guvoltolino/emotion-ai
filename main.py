import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import ChatPromptTemplate
import streamlit as st
import speech_recognition as sr

load_dotenv()

api_model = os.getenv("GOOGLE_API_MODEL")
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    temperature=0.2,
    google_api_key=api_key,
    model=api_model,
    language="pt"
)

def detect_emotion(text):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um modelo de linguagem treinado para detectar emoções em textos."),
        ("user", f"Detecte se a emoção no seguinte texto é positiva, negativa ou neutra: '{text}'")
    ])
    prompt_text = prompt.format_prompt(text=text)
    response = llm.invoke(prompt_text)
    return response.content.strip()

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.write("Aguardando fala...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio, language="pt-BR")
        st.write(f"Você disse: {text}")
        return text
    except sr.UnknownValueError:
        st.write("Não consegui entender o que foi dito.")
        return None
    except sr.RequestError:
        st.write("Erro ao solicitar resultados do Google Speech Recognition.")
        return None

# Configurações de layout
st.set_page_config(
    page_title="EmotiSense - Detector de Emoções",
    page_icon=":speech_balloon:",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #004080;
        color: #ffffff;
        padding: 20px;
    }
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-1q8dd3e {
        color: #ffffff;
    }
    .sidebar-title {
        font-size: 30px;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .sidebar-text {
        font-size: 16px;
        color: #ffffff;
        text-align: justify;
        margin-bottom: 20px;
    }
    .stButton {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<p class="sidebar-title">EmotiSense</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-text">Detecte as emoções em suas falas ou textos com o EmotiSense. Descubra se suas emoções são positivas, negativas ou neutras.</p>', unsafe_allow_html=True)

user_avatar = "🙂"  # Emoji mais neutro e amigável
assistant_avatar = "🤖"  # Emoji de assistente robótico

st.title("EmotiSense - Detector de Emoções")
st.markdown("***Descubra se suas emoções são positivas, negativas ou neutras!***")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Bem-vindo ao EmotiSense! Como posso ajudar a identificar emoções hoje?"}
    ]

for message in st.session_state["messages"]:
    if message["role"] != "system":
        avatar = assistant_avatar if message["role"] == "assistant" else user_avatar
        st.chat_message(message["role"], avatar=avatar).write(message["content"])

if st.button("Falar", key="speak_button"):
    text = recognize_speech_from_mic()
    if text:
        st.session_state["messages"].append({"role": "user", "content": text})
        st.chat_message("user", avatar=user_avatar).write(text)
        emotion = detect_emotion(text)
        st.session_state["messages"].append({"role": "assistant", "content": f"A emoção detectada é: {emotion}"})
        st.chat_message("assistant", avatar=assistant_avatar).write(f"A emoção detectada é: {emotion}")

if prompt := st.chat_input("Digite sua pergunta aqui."):
    prompt = prompt.strip().lower()

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        st.chat_message("user", avatar=user_avatar).write(prompt)

        with st.spinner("Processando..."):
            emotion = detect_emotion(prompt)
            st.session_state["messages"].append({"role": "assistant", "content": f"A emoção detectada é: {emotion}"})
            st.chat_message("assistant", avatar=assistant_avatar).write(f"A emoção detectada é: {emotion}")

    else:
        st.write("Por favor, digite uma pergunta.")
