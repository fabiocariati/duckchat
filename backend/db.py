import os
import re
import tempfile
from pathlib import Path
import duckdb
from typing import Dict, List, Tuple, Any


class DuckDBController:
    SUPPORTED_FILE_TYPES = ["csv", "parquet"]

    def __init__(self, database=':memory:', read_only=False):
        self.con = duckdb.connect(database=database, read_only=read_only)
        self.tables: Dict[str, Dict[str, Any]] = {}

    def register_file_as_table(self, file) -> str:
        """Register an uploaded file as a DuckDB table."""
        suffix = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.seek(0)
            tmp.write(file.read())
            tmp_path = tmp.name

        table_name = self._filename_to_table_name(file.name)
        source_fn_mapper = {".csv": "read_csv_auto",
                            ".parquet": "read_parquet"}

        try:
            if suffix not in [f".{ext}" for ext in ["csv", "parquet"]]:
                raise ValueError(
                    f"Unsupported file format: {suffix}. Supported files: csv, parquet"
                )
            self.con.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS 
                SELECT * FROM {source_fn_mapper.get(suffix)}('{tmp_path}')
            """)
        finally:
            os.unlink(tmp_path)

        # Store table metadata
        self.tables[file.name] = {
            "table_name": table_name,
            "columns": [n[1] for n in self.get_table_schema(table_name)],
            "schema": self.get_table_schema(table_name),
            "ddl": self.get_table_ddl(table_name),
            "sample": self.get_table_sample(table_name)
        }

        return table_name

    def get_table_schema(self, table_name: str) -> List[Tuple]:
        """Get the schema of a table."""
        schema_result = self.con.execute(
            f"PRAGMA table_info('{table_name}')").fetchall()
        if not schema_result:
            raise Exception(
                f"Could not retrieve schema for table '{table_name}'. Please check the file."
            )
        return schema_result

    def get_table_ddl(self, table_name: str) -> str:
        """Get the DDL (CREATE TABLE statement) for a table."""
        result = self.con.execute(f"PRAGMA table_info('{table_name}')")
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

    def get_table_sample(self, table_name: str, sample_size: int = 3) -> str:
        """Get a sample of rows from a table."""
        result = self.con.execute(
            f"SELECT * FROM {table_name} USING SAMPLE {sample_size} ROWS")
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        if not rows:
            raise Exception(
                f"Could not retrieve sample from table '{table_name}'. Please check the file.")
        csv_lines = [",".join(columns)]
        for row in rows:
            row_str = ",".join(map(str, row))
            csv_lines.append(row_str)
        return "\n".join(csv_lines)

    def execute_query(self, sql: str) -> Any:
        """Execute a SQL query and return the result."""
        return self.con.execute(sql).fetchdf()

    @staticmethod
    def _filename_to_table_name(filename: str) -> str:
        """Convert a filename to a valid table name."""
        name, _ = os.path.splitext(filename)
        name = re.sub(r'\W+', '_', name)
        if name[0].isdigit():
            name = f"file_{name}"
        return name

    def close(self):
        """Close the DuckDB connection."""
        self.con.close()
