import os
import re
import tempfile
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


def get_table_ddl(con, table_name):
    result = con.execute(f"PRAGMA table_info('{table_name}')")
    rows = result.fetchall()
    columns = result.description

    if not rows:
        raise ValueError(f"Table '{table_name}' not found.")

    col_index = {desc[0]: idx for idx, desc in enumerate(columns)}
    col_defs = []
    for row in rows:
        name = row[col_index['name']]
        col_type = row[col_index['type']]
        col_def = f"{name} {col_type}"

        if row[col_index['notnull']]:
            col_def += " NOT NULL"
        if default := row[col_index['dflt_value']]:
            col_def += f" DEFAULT {default}"
        col_defs.append(col_def)

    return f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(col_defs) + "\n);"


def filename_to_table_name(filename):
    name, _ = os.path.splitext(filename)
    name = re.sub(r'\W+', '_', name)
    if name[0].isdigit():
        name = f"file_{name}"
    return name


def register_file_as_table(con, file):
    suffix = Path(file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.seek(0)
        tmp.write(file.read())
        tmp_path = tmp.name

    table_name = filename_to_table_name(file.name)
    source_fn_mapper = {".csv": "read_csv_auto", ".parquet": "read_parquet"}

    try:
        if suffix not in [f".{ext}" for ext in supported_files]:
            raise ValueError(
                f"Unsupported file format: {suffix}. Supported files: {str(supported_files)}"
            )
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS 
            SELECT * FROM {source_fn_mapper.get(suffix)}('{tmp_path}')
        """)
    finally:
        os.unlink(tmp_path)

    return table_name


def get_table_schema(con, table_name):
    schema_result = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    if not schema_result:
        raise Exception(
            f"Could not retrieve schema for table '{table_name}'. Please check the file."
        )
    return schema_result


def get_table_sample(con, table_name, sample_size=3):
    result = con.execute(f"SELECT * FROM {table_name} USING SAMPLE {sample_size} ROWS")
    columns = [desc[0] for desc in result.description]
    rows = result.fetchall()
    if not rows:
        raise Exception(f"Could not retrieve sample from table '{table_name}'. Please check the file.")
    csv_lines = [",".join(columns)]
    for row in rows:
        row_str = ",".join(map(str, row))
        csv_lines.append(row_str)
    return "\n".join(csv_lines)


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
