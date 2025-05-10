import re
import sqlglot
from langchain_core.output_parsers import BaseOutputParser


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

        pattern = re.compile(
            rf"^\s*({'|'.join(valid_commands)})\b", re.IGNORECASE)
        if not pattern.match(query):
            raise ValueError(
                f"Output does not look like a valid SQL query: {query}")

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
