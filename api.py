from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import duckdb
import tempfile
import os

from duckchat import (
    ModelProviderController,
    SQLGenerator,
    DuckDBController
)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize controllers
db_controller = DuckDBController(database=':memory:', read_only=False)
provider_controller = ModelProviderController()
sql_generator = SQLGenerator(provider_controller)

class QueryRequest(BaseModel):
    prompt: str
    provider: str
    model: str
    api_key: Optional[str] = None

class QueryResponse(BaseModel):
    sql: str
    dataframe: List[Dict[str, Any]]

@app.get("/providers")
async def get_providers():
    return {"providers": provider_controller.get_providers()}

@app.get("/models/{provider}")
async def get_models(provider: str):
    provider_controller.provider = provider
    return {"models": provider_controller.get_models()}

@app.post("/query")
async def execute_query(request: QueryRequest):
    try:
        # Configure provider
        provider_controller.provider = request.provider
        provider_controller.add_parameter("model", request.model)
        if request.api_key:
            provider_controller.add_parameter("api_key", request.api_key)

        # Generate and execute SQL
        generated_sql = sql_generator.generate_sql(request.prompt, db_controller.tables)
        result_df = db_controller.execute_query(generated_sql)
        
        # Convert DataFrame to list of dicts for JSON serialization
        result_data = result_df.to_dict(orient='records')
        
        return {
            "sql": generated_sql,
            "dataframe": result_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Create a temporary file that mimics a file-like object
        # with a .name attribute as expected by register_file_as_table
        class TempFileWithName:
            def __init__(self, name, content):
                self.name = name
                self.content = content
                
            def read(self):
                return self.content
                
            def seek(self, position):
                pass  # no-op for our case
        
        # Read content once
        content = await file.read()
        
        # Create our file-like object with the original filename
        temp_file = TempFileWithName(file.filename, content)
        
        # Register the file as a table
        table_name = db_controller.register_file_as_table(temp_file)
        
        return {
            "table_name": table_name,
            "columns": db_controller.tables[file.filename]["columns"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tables")
async def get_tables():
    return {"tables": db_controller.tables}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 