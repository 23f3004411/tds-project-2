import os, sys
import traceback

import io
import json
import base64
import uvicorn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from bs4 import BeautifulSoup
import requests

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain import hub

load_dotenv()

app = FastAPI(
    title="Data Analyst Agent API",
    description="An agent that uses LLMs to source, prepare, analyze, and visualize data.",
    version="1.6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

@tool
def scrape_webpage(url: str) -> str:
    """
    Scrapes the text content from a given URL.
    Use this tool to get information from a webpage to answer questions.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url.strip(), headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join(soup.get_text().split())
    except requests.RequestException as e:
        return f"Error scraping URL {url}: {e}"

CODE_GEN_PROMPT = PromptTemplate.from_template(
"""You are an expert Python data analyst. A pandas DataFrame named `df` has been pre-loaded in memory.

** DATA IF NOT AVAIABLE USE THIS: (RAW FILE)**
{df_json_str}

**DataFrame Structure:**
{df_structure}

Your task is to write a Python script to answer the question: {question}

**VERY IMPORTANT:**
- DO NOT try to load any data or read any files (e.g., do not use `pd.read_csv`).
- You MUST use the existing DataFrame named `df` with the structure provided above.

**Instructions:**
1.  Use the pandas, matplotlib, and networkx libraries.
2.  The final output MUST be stored in a variable `result`.
3.  The `result` must be the final answer (e.g., number, string, list, or a base64 plot).
4.  If plotting, the `result` MUST be a base64 encoded data URI string.
5.  The script must be a direct sequence of commands, not a function.
6.  Return ONLY the raw Python code. Do not use markdown fences or add explanations.

**Plotting Example:**
import io, base64, matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(df['col_a'], df['col_b'])
plt.xlabel('Column A'); plt.ylabel('Column B'); plt.title('Scatter Plot')
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
result = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
plt.close()

**Non-Plotting Example:**
result = len(df)
"""
)

@tool
def analyze_dataframe(question: str, df_json_string: str) -> str:
    """
    Analyzes a pandas DataFrame to answer a question by generating and executing Python code.
    - question: The question or task to perform on the DataFrame.
    - df_json_string: The DataFrame in JSON format with 'split' orientation.
    """
    python_code = ""
    try:
        df = pd.read_json(io.StringIO(df_json_string), orient='split')
        
        if df.empty:
            return "Error: The provided data is empty."

        df_head = df.head().to_string()
        df_structure = f"Columns: {df.columns.tolist()}\n\nFirst 5 rows:\n{df_head}"

        code_generation_prompt = CODE_GEN_PROMPT.format(
            question=question,
            df_structure=df_structure,
            df_json_str = df_json_string
        )
        response = llm.invoke(code_generation_prompt)
        python_code = response.content.strip()

        local_vars = {'df': df}
        exec_globals = {
            'pd': pd, 'np': np, 'plt': plt,
            'io': io, 'base64': base64, 'nx': nx
        }

        exec(python_code, exec_globals, local_vars)

        result = local_vars.get('result')
        if result is None:
            return "Error: The generated code did not produce a 'result' variable. This might be due to an error in the code that was not caught."
        
        if isinstance(result, (pd.Series, pd.DataFrame)):
            return result.to_json(orient='split')
        if isinstance(result, np.generic):
            return str(result.item())
        return str(result)

    except Exception as e:
        print(traceback.format_exc())
        return f"An error occurred while executing the generated Python code. Please analyze this error and try again.\n--- ERROR ---\n{type(e).__name__}: {e}\n--- ATTEMPTED CODE ---\n{python_code}"

@app.post("/api/")
async def data_analyst_endpoint(
    questions_file: UploadFile = File(..., alias="questions.txt"),
    attachments: Optional[List[UploadFile]] = File(None)
):
    questions_text = (await questions_file.read()).decode('utf-8')
    
    df_json_str = ""
    if attachments:
        attachment = attachments[0]
        file_content = await attachment.read()
        try:
            if attachment.filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
                df_json_str = df.to_json(orient='split')
            elif attachment.filename.endswith('.json'):
                df_json_str = file_content.decode('utf-8')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file {attachment.filename}: {e}")

    tools_for_this_request = [scrape_webpage]

    if df_json_str:
        @tool
        def analyze_provided_data(question: str) -> str:
            """Use this tool ONLY to answer a question about the provided data file. The input for this tool is the user's question."""
            return analyze_dataframe.invoke({
                "question": question,
                "df_json_string": df_json_str
            })
        
        tools_for_this_request.append(analyze_provided_data)

    prompt = hub.pull("hwchase17/react")

    # The prompt object has a 'template' attribute which holds the string
    original_template = prompt.template

    # Define your custom lines to be added
    custom_lines = """
    Final Answer formating:
    Your final answer MUST be a single, valid JSON array containing the answers to all questions, in the exact order they were asked.
    - If an answer is a number, use the number type (e.g., 42 or 0.485).
    - If an answer is a string, enclose it in double quotes (e.g., "Titanic").
    - If an answer is a plot, return only the base64 data URI string.
    - Do not add any explanatory text outside of the JSON array.
    
    Additional Info: If Possible to give a numeric answer, give a numeric digit only.
    For example: Count ofs questions, etc
    Also compress any image generated.
    """

    # Combine the original template with your custom lines
    new_template = original_template + "\n" + custom_lines

    # Create a new PromptTemplate instance from the modified string
    prompt = PromptTemplate.from_template(new_template)
    
    agent = create_react_agent(llm, tools_for_this_request, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_this_request,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    agent_input = {"input": questions_text}
    
    try:
        response = agent_executor.invoke(agent_input)
        output_str = response.get('output', '{}')
        
        try:
            return json.loads(output_str)
        except json.JSONDecodeError:
            return {"answer": output_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during agent execution: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
