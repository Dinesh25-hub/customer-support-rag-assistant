import requests
import streamlit as st

from llm import stream_answer

st.set_page_config(page_title="Support Bot", page_icon="🤖")
st.title("Support Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_answer(st.session_state.messages))
        except requests.RequestException:
            reply = "⚠️ Could not reach Ollama. Is `ollama serve` running?"
            st.error(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})