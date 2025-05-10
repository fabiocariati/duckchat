from typing import Dict, Any
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from model_provider import ModelProviderController
from sql_parser import SQLOutputParser


class SQLGenerator:
    def __init__(self, provider_controller: ModelProviderController):
        self.provider_controller = provider_controller
        self.system_template = """
            You are an assistant that writes DuckDB-compatible SQL queries.
            Your output always must contains a valid SQL query.
            Your output must contain the generated SQL between ```sql and ```.
            Inside ```sql and ``` must be only the generated SQL, without explanations, comments, or any other additional text.
            If the table name is not mentioned in the user prompt, consider the available tables, schemas, and examples (provided below) to find out which table contains the information.

            Available tables and schema:
            -----------------
            {tables_info}
        """

    def _format_tables_info(self, tables: Dict[str, Dict[str, Any]]) -> str:
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
        return tables_info

    def generate_sql(self, user_prompt: str, tables: Dict[str, Dict[str, Any]]) -> str:
        """Generate SQL query from user prompt using the available tables."""
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_template),
            HumanMessagePromptTemplate.from_template(f"{user_prompt} Respond with SQL")
        ])
        
        input_vars = {
            "tables_info": self._format_tables_info(tables),
            "user_prompt": user_prompt,
        }
        
        chain = prompt_template | self.provider_controller.get_chat() | SQLOutputParser()
        generated_sql = chain.invoke(input_vars)
        
        if not generated_sql.lower().startswith(("select", "with")):
            raise Exception(f"Generated SQL seems invalid: {generated_sql}")
            
        return generated_sql

