import os
import re
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import ollama
import sqlglot

load_dotenv()
supported_files = ["csv", "parquet"]
openai_api_key = os.getenv("OPENAI_API_KEY")


class ModelProviderController:
    PROVIDERS = {
        "ollama": {
            "models": [m.model for m in ollama.list().models],
            "chat_class": ChatOllama,
            "params": {"base_url": os.getenv("OLLAMA_HOST")},
        },
        "openai": {
            "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4-turbo" ],
            "chat_class": ChatOpenAI,
            "params": {},
        },
    }

    def __init__(self):
        self.provider = "ollama"

    def add_parameter(self, key, value):
        self.PROVIDERS[self.provider]["params"][key] = value

    def get_providers(self):
        return list(self.PROVIDERS.keys())

    def get_chat(self):
        provider = self.PROVIDERS[self.provider]
        if provider == "openai" and not provider["params"].get("api_key"):
            raise ValueError("API key is required for OpenAI provider. Please set it.")
        return provider["chat_class"](**provider["params"], temperature=0.1)
    
    def get_models(self):
        return self.PROVIDERS[self.provider]["models"]


class SQLOutputParser(BaseOutputParser):
    def parse(self, text):
        match = re.search(r"```sql([\s\S]*?)```", text)
        if match:
            text = match.group(1)

        valid_commands = (
            "SELECT", "UPDATE", "INSERT", "DELETE", "CREATE", "ALTER", "DROP",
            "TRUNCATE", "BEGIN", "COMMIT", "ROLLBACK", "WITH"
        )
        query = self._clear_sql(text, valid_commands)

        if not query:
            raise ValueError("No SQL query output provided.")

        pattern = re.compile(rf"^\s*({'|'.join(valid_commands)})\b", re.IGNORECASE)
        if not pattern.match(query):
            raise ValueError(f"Output does not look like a valid SQL query: {query}")

        return sqlglot.transpile(query, write="duckdb", pretty=True)[0]

    @staticmethod
    def _clear_sql(sql, valid_commands):
        clean_sql = re.sub(r'```(?:sql)?\s*([\s\S]*?)```', r'\1', sql)
        clean_sql = re.sub(
            r'SQL:|Query:|SQL Query:|Output:|Output Query:',
            '',
            clean_sql,
            flags=re.IGNORECASE,
        )
        clean_sql = clean_sql.strip().rstrip(";").strip()

        pattern = re.compile(rf"{'|'.join(valid_commands)}", re.IGNORECASE)
        match = pattern.search(clean_sql)
        if match:
            return clean_sql[match.start():].strip()
        return clean_sql


def prompt_to_sql(user_prompt, tables, provider_controller: ModelProviderController):
    system_template = """
        You are an assistant that writes DuckDB-compatible SQL queries.
        Your output always must contains a valid SQL query.
        Your output must contain the generated SQL between ```sql and ```.
        Inside ```sql and ``` must be only the generated SQL, without explanations, comments, or any other additional text.
        If the table name is not mentioned in the user prompt, consider the available tables, schemas, and examples (provided below) to find out which table contains the information.

        Available tables and schema:
        -----------------
        {tables_info}
    """

    tables_info = ""
    for table in tables.values():
        tables_info += f"""
            Table Name: `{table['table_name']}`
            DDL:
            ```sql
            {table['ddl']}
            ```
            Sample Rows:
            ```csv
            {table['sample']}
            ```
        """

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(f"{user_prompt} Respond with SQL")
    ])
    input_vars = {
        "tables_info": tables_info,
        "user_prompt": user_prompt,
    }
    chain = prompt_template | provider_controller.get_chat() | SQLOutputParser()

    generated_sql = chain.invoke(input_vars)
    if not generated_sql.lower().startswith(("select", "with")):
        raise Exception(f"Generated SQL seems invalid: {generated_sql}")
    return generated_sql
