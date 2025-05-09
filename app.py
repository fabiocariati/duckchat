import streamlit as st
from streamlit_ace import st_ace
import requests
import json
from typing import Dict, Any

API_BASE_URL = "http://api:8000"

def close_edit_chat():
    st.session_state["edit_index"] = None
    st.session_state["edit_text"] = ""
    st.rerun()

def render_assistent_message(i, msg):
    if msg.get("type") == "error":
        st.error(msg["content"])
    else:
        st.markdown(msg["content"])
    if msg.get("sql"):
        if st.session_state["edit_index"] != i:
            st.code(msg["sql"], language="sql")
        if st.session_state["edit_index"] == i:
            edited_sql = st_ace(
                value=msg["sql"],
                language="sql",
                theme="xcode",
                key=f"edit_sql_{i}",
                height=200,
                min_lines=10,
                auto_update=True
            )
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Confirm SQL edit", key=f"confirm_sql_edit_{i}"):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/query",
                            json={
                                "prompt": edited_sql,
                                "provider": st.session_state["provider"],
                                "model": st.session_state["model"],
                                "api_key": st.session_state.get("api_key")
                            }
                        )
                        response.raise_for_status()
                        result = response.json()
                        st.session_state["messages"] = st.session_state["messages"][:i]
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": "Result of the edited SQL:",
                            "sql": result["sql"],
                            "dataframe": result["dataframe"]
                        })
                        close_edit_chat()
                    except Exception as e:
                        st.error(f"Error executing edited SQL: {e}")
            with col2:
                if st.button("Cancel edit", key=f"cancel_sql_edit_{i}"):
                    close_edit_chat()
        elif st.session_state["edit_index"] is None:
            if st.button("🛠️ Edit SQL", key=f"edit_sql_btn_{i}"):
                st.session_state["edit_index"] = i
                st.rerun()
    if msg.get("dataframe") is not None:
        st.dataframe(msg["dataframe"])

def render_user_message(i, msg):
    if st.session_state["edit_index"] == i:
        idx = st.session_state["edit_index"]
        new_input = st.text_input(
            "Edit message:",
            value=st.session_state["edit_text"],
            key=f"edit_input_{idx}"
        )
        if st.button("Confirm edit", key=f"confirm_edit_{i}"):
            st.session_state["messages"] = st.session_state["messages"][:i]
            st.session_state["rerun_prompt"] = new_input
            close_edit_chat()
    elif st.session_state["edit_index"] is None:
        if st.button("✏️ Edit", key=f"edit_btn_{i}"):
            st.session_state["edit_index"] = i
            st.session_state["edit_text"] = msg["content"]
            st.rerun()
        st.markdown(msg["content"])

def render_chat():
    for i, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                render_user_message(i, msg)
            elif msg["role"] == "assistant":
                render_assistent_message(i, msg)

    prompt = st.chat_input("Enter your data query") or st.session_state.pop("rerun_prompt", None)
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state["tables"]:
            st.warning("Please upload at least one file.", icon="⚠️")
            return

        with st.chat_message("assistant"):
            with st.spinner(f"🧠 Generating SQL"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        json={
                            "prompt": prompt,
                            "provider": st.session_state["provider"],
                            "model": st.session_state["model"],
                            "api_key": st.session_state.get("api_key")
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    st.code(result["sql"], language="sql")
                    st.dataframe(result["dataframe"])

                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": "Query and the result:",
                            "sql": result["sql"],
                            "dataframe": result["dataframe"]
                        }
                    )
                except Exception as e:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": f"Error: {e}", "type": "error"}
                    )
                finally:
                    st.rerun()

def render_side_bar():
    st.sidebar.header("⚙️ Settings")
    
    # Get available providers
    providers_response = requests.get(f"{API_BASE_URL}/providers")
    providers = providers_response.json()["providers"]
    
    st.session_state["provider"] = st.sidebar.selectbox("Provider", providers, index=0)
    
    # Get available models for selected provider
    models_response = requests.get(f"{API_BASE_URL}/models/{st.session_state['provider']}")
    models = models_response.json()["models"]
    
    st.session_state["model"] = st.sidebar.selectbox("Model", options=models, index=0)
    
    if st.session_state["provider"] == "openai":
        st.session_state["api_key"] = st.sidebar.text_input("API Key", type="password")
    elif st.session_state["provider"] == "ollama" and models == []:
        st.sidebar.error("No models found. Please add a model: `docker exec -it ollama ollama pull llama3:8b`")

    st.sidebar.header("📂 Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more CSV or Parquet files",
        type=["csv", "parquet"],
        accept_multiple_files=True
    )

    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state["tables"]:
            with st.spinner(f"Registering '{uploaded_file.name}'..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{API_BASE_URL}/upload", files=files)
                response.raise_for_status()
                result = response.json()
                st.session_state["tables"][uploaded_file.name] = {
                    "table_name": result["table_name"],
                    "columns": result["columns"]
                }

    st.sidebar.title("📊 Tables")
    for filename, details in st.session_state["tables"].items():
        with st.sidebar.expander(filename, expanded=False):
            for col in details["columns"]:
                st.markdown(f"- {col}")

def session_state_defaults(defaults):
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_app():
    session_state_defaults({
        "edit_index": None,
        "edit_text": "",
        "messages": [],
        "rerun_prompt": None,
        "tables": {},
        "provider": None,
        "model": None,
        "api_key": None
    })

    st.set_page_config(page_title="Duckchat", layout="wide")

    render_side_bar()
    render_chat()

if __name__ == "__main__":
    render_app()
