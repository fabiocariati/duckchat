import duckdb
import streamlit as st
from streamlit_ace import st_ace


from duckchat.utils import (
    ModelProviderController,
    prompt_to_sql,
    register_file_as_table,
    get_table_schema,
    get_table_ddl,
    get_table_sample,
    supported_files,
    openai_api_key
)


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
                        con = st.session_state["con"]
                        new_df = con.execute(edited_sql).fetchdf()
                        st.session_state["messages"] = st.session_state["messages"][:i]
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": "Result of the edited SQL:",
                            "sql": edited_sql,
                            "dataframe": new_df
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
    provider_controller = st.session_state["model_provider_controller"]

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
                    tables = st.session_state["tables"]
                    generated_sql = prompt_to_sql(prompt, tables, provider_controller)
                    st.code(generated_sql, language="sql")

                    with st.spinner("⚡ Executing SQL query..."):
                        result_df = st.session_state["con"].execute(generated_sql).fetchdf()
                        st.dataframe(result_df)

                    st.session_state["messages"].append(
                        {"role": "assistant", "content": "Query and the result:", "sql": generated_sql,
                         "dataframe": result_df}
                    )
                except Exception as e:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": f"Error: {e}", "type": "error"}
                    )
                finally:
                    st.rerun()


def render_side_bar():
    st.sidebar.header("⚙️ Settings")
    provider_controller = st.session_state["model_provider_controller"]
    provider_controller.provider = st.sidebar.selectbox("Provider", provider_controller.get_providers(), index=0)
    model = st.sidebar.selectbox("Model", options=provider_controller.get_models(), index=0)
    provider_controller.add_parameter("model", model)
    if provider_controller.provider == "openai":
        api_key = st.sidebar.text_input("API Key", value=openai_api_key, type="password")
        provider_controller.add_parameter("api_key", api_key)
    elif provider_controller.provider == "ollama" and provider_controller.get_models() == []:
        st.sidebar.error("No models found. Please add a model: `docker exec -it ollama ollama pull llama3:8b`")

    st.sidebar.header("📂 Files")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more CSV or Parquet files",
        type=supported_files,
        accept_multiple_files=True
    )
    con = st.session_state["con"]
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state["tables"]:
            with st.spinner(f"Registering '{uploaded_file.name}'..."):
                table_name = register_file_as_table(con, uploaded_file)
                table_schema = get_table_schema (con, table_name)
                st.session_state["tables"][uploaded_file.name] = {
                    "table_name": table_name,
                    "columns": [n[1] for n in table_schema],
                    "schema": table_schema,
                    "ddl": get_table_ddl(con, table_name),
                    "sample": get_table_sample(con, table_name)
                }

    st.sidebar.title("📊 Tables")
    for _, details in st.session_state["tables"].items():
        with st.sidebar.expander(details.get("table_name"), expanded=False):
            for col in details.get("columns", []):
                st.markdown(f"- {col}")


def session_state_defaults(defaults):
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_app():
    session_state_defaults({
        "con": duckdb.connect(database=':memory:', read_only=False),
        "model_provider_controller": ModelProviderController(),
        "edit_index": None,
        "edit_text": "",
        "messages": [],
        "rerun_prompt": None,
        "tables": {}
    })

    st.set_page_config(page_title="Duckchat", layout="wide")

    render_side_bar()
    render_chat()


if __name__ == "__main__":
    render_app()
