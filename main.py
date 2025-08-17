import io
import os
import re
import json
import base64
import traceback
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, Response
from flask_cors import CORS 
from werkzeug.utils import secure_filename

from dotenv import load_dotenv

import requests  
import pandas as pd
import numpy as np
import duckdb
from bs4 import BeautifulSoup

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from openai import OpenAI

import logging
from logging import StreamHandler, Formatter

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
handler = StreamHandler()
handler.setFormatter(Formatter(fmt="%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

def _preview(label: str, obj, limit=500):
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    s = s if len(s) <= limit else s[:limit] + "... [truncated]"
    logger.info("%s: %s", label, s)

load_dotenv()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  
MAX_IMAGE_BYTES = 6_500  
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/agent_uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

CORS(app, resources={
    r"/api/": {
        "origins": "*",        
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False,
        "max_age": 600
    }
})

logger.info("CORS enabled for /api/ with origins=*, methods=POST,OPTIONS")

def fig_to_data_uri_png(fig, max_bytes=MAX_IMAGE_BYTES):
    def encode(dpi):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        return data_uri
    for dpi in (120, 96, 80, 72, 60, 45, 30, 20, 15):
        uri = encode(dpi)
        if len(uri.encode("utf-8")) <= max_bytes:
            return uri
    raise ValueError("Image exceeds size budget; simplify plot or reduce points.")

def tool_web_fetch(url: str) -> Dict[str, Any]:
    """
    Fetch a web page and return:
    {
      "url": str,
      "status": int,
      "html": str,     
      "tables": List[List[Dict[str, Any]]],  
      "title": str
    }
    """
    logger.info("[tool:web_fetch] GET %s", url)
    try:
        resp = requests.get(url, timeout=20)
        logger.info("[tool:web_fetch] status=%s length=%s", resp.status_code, len(resp.text or ""))
        status = resp.status_code
        html = resp.text if resp.ok else ""
        title = ""
        tables = []

        if html:
            soup = BeautifulSoup(html, "html.parser")

            title_tag = soup.find("title")
            title = title_tag.text.strip() if title_tag else ""

            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "meta", "link"]):
                tag.decompose()

            clean_text = soup.get_text(separator="\n", strip=True)

            try:
                dfs = pd.read_html(html)
                logger.info("[tool:web_fetch] parsed %d tables", len(dfs))
                for df in dfs:
                    tables.append(json.loads(df.to_json(orient="records")))
            except ValueError:
                logger.info("[tool:web_fetch] no HTML tables found")

            html = clean_text

        return {"url": url, "status": status, "html": html, "tables": tables, "title": title}

    except Exception as e:
        logger.exception("[tool:web_fetch] error: %s", e)
        return {"url": url, "status": 0, "error": str(e), "html": "", "tables": [], "title": ""}

def tool_run_code(code: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute user/LLM generated Python code in a constrained namespace.
    Provides pandas, numpy, duckdb, matplotlib.pyplot as plt, seaborn as sns, and a dict 'inputs'.
    Returns {"stdout": str, "vars": dict (json-serializable best-effort), "error": str|None}.
    """
    _preview("[tool:run_code] inputs", list(inputs.keys()) if isinstance(inputs, dict) else type(inputs).__name__)
    _preview("[tool:run_code] code", code, limit=800)
    allowed_globals = {
        "pd": pd,
        "np": np,
        "duckdb": duckdb,
        "plt": plt,
        "sns": sns,
        "base64": base64,
        "io": io,
        "re": re,
        "inputs": inputs,
    }
    local_vars: Dict[str, Any] = {}
    stdout_capture = io.StringIO()
    try:
        import contextlib, sys
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, allowed_globals, local_vars)
        def safe(obj):
            try:
                json.dumps(obj)
                return obj
            except TypeError:
                return str(obj)
        serializable_vars = {k: safe(v) for k, v in local_vars.items() if not k.startswith("_")}
        return {"stdout": stdout_capture.getvalue(), "vars": serializable_vars, "error": None}
    except Exception as e:
        logger.exception("[tool:run_code] error")
        return {"stdout": stdout_capture.getvalue(), "vars": {}, "error": f"{e}\n{traceback.format_exc()}"}

def tool_plot_png(code: str) -> Dict[str, Any]:
    """
    Execute plotting code that creates a Matplotlib figure and returns a base64 data URI PNG.
    The code should end with 'fig = plt.gcf()' or create a figure assigned to 'fig'.
    """
    _preview("[tool:plot_png] code", code, limit=800)
    allowed_globals = {
        "pd": pd,
        "np": np,
        "base64": base64,
        "plt": plt,
        "re": re,
        "sns": sns,
        "io": io,
    }
    local_vars: Dict[str, Any] = {}
    stdout_capture = io.StringIO()
    try:
        import contextlib
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, allowed_globals, local_vars)
        fig = local_vars.get("fig", plt.gcf())
        if not fig:
            raise ValueError("No figure found; ensure code assigns 'fig = plt.gcf()' or creates a figure.")
        uri = fig_to_data_uri_png(fig, MAX_IMAGE_BYTES)
        logger.info("[tool:plot_png] data_uri_size=%d bytes", len(uri.encode("utf-8")))
        return {"data_uri": uri, "stdout": stdout_capture.getvalue(), "error": None}
    except Exception as e:
        logger.exception("[tool:plot_png] error")
        return {"data_uri": None, "stdout": stdout_capture.getvalue(), "error": f"{e}\n{traceback.format_exc()}"}

def get_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "name": "web_fetch",
            "description": "Fetch a webpage by URL, parse title and HTML tables where possible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "HTTP/HTTPS URL to fetch"}
                },
                "required": ["url"],
            },
        },
        {
            "name": "run_code",
            "description": "Execute Python code for data analysis using pandas/numpy/duckdb. Access uploaded file paths via provided 'inputs'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "inputs": {"type": "object", "description": "Named inputs such as file paths or small data"},
                },
                "required": ["code", "inputs"],
            },
        },
        {
            "name": "plot_png",
            "description": "Generate a Matplotlib plot and return a base64 data URI under 100,000 bytes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code that builds a plot and sets 'fig' variable"},
                },
                "required": ["code"],
            },
        },
    ]

def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "web_fetch":
        return tool_web_fetch(**arguments)
    if name == "run_code":
        return tool_run_code(**arguments)
    if name == "plot_png":
        return tool_plot_png(**arguments)
    return {"error": f"Unknown tool: {name}"}

SYSTEM_PROMPT = (
    "You are an agentic AI that reads a questions.txt, optional extra files, "
    "and uses tools to fetch web data, analyze with Python, and plot. "
    "Always produce exactly the final answer format requested in questions.txt "
    "(for example, a JSON array or object). Never hard-code answers; compute them. "
    "When returning plots, ensure data URIs are PNG and stay under 100,000 bytes."
    "Use only pandas as pd, numpy as np, duckdb and duckdb, matplotlib as plt, seaborn as sns in python. NO OTHER LIBRARY! No need To convert anything to Base64 yourself, just give a plot"
    "Check varaiable names!"
)

def run_agent(questions_text: str, file_index: Dict[str, str]) -> str:
    """
    Orchestrates a Chat Completions loop with function calling until a final answer is produced.
    questions_text: content of questions.txt
    file_index: map of uploaded filename -> saved path on disk
    Returns final model text.
    """
    tools = get_tool_schemas()
    _preview("[agent] questions.txt", questions_text, limit=1200)
    _preview("[agent] uploaded_files", file_index)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Here is questions.txt followed by an index of uploaded files.\n\n"
            f"questions.txt:\n{questions_text}\n\n"
            f"uploaded_files:\n{json.dumps(file_index, indent=2)}\n"
            "Use tools as needed. If code is required, write it and run it via run_code. "
            "For plots, use plot_png and return the data URI."
        )},
    ]
    turn = 0
    while True:
        turn += 1
        logger.info("[agent] requesting completion (turn=%d)", turn)
        _preview("[agent] messages->", messages, limit=1200)

        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=[{"type": "function", "function": t} for t in tools],
            tool_choice="auto",
            temperature=0.0,
        )

        msg = completion.choices[0].message

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                for tc in tool_calls],
            })

            for tc in tool_calls:
                name = tc.function.name
                args = {}
                if tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                if name == "run_code":
                    args.setdefault("inputs", {})
                    args["inputs"].setdefault("files", file_index)

                result = call_tool(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                })

            continue

        final_text = msg.content or ""
        return final_text

def try_parse_json(s: str):
    try:
        return True, json.loads(s)
    except Exception:
        return False, None

def clamp_size_json(obj, max_bytes=150_000):
    """Heuristic: try to reduce precision of floats and shorten strings."""
    def shrink(x):
        if isinstance(x, float):
            return float(f"{x:.6g}") 
        if isinstance(x, str) and len(x) > 2000:
            return x[:2000] + "...[truncated]"
        if isinstance(x, list):
            return [shrink(v) for v in x]
        if isinstance(x, dict):
            return {k: shrink(v) for k, v in x.items()}
        return x
    shrunk = shrink(obj)
    data = json.dumps(shrunk, ensure_ascii=False)
    if len(data.encode("utf-8")) <= max_bytes:
        return True, data
    return False, data

@app.route("/api/", methods=["POST"])
def api():
    for i in request.files:
        logger.info("[file name]: " + i)
    if "questions.txt" not in request.files:
        print("here")
        return jsonify({"error": "questions.txt file is required in multipart/form-data"}), 400

    saved_index: Dict[str, str] = {}
    for key, storage in request.files.items():
        filename = secure_filename(storage.filename)
        if not filename:
            continue
        save_path = os.path.join(UPLOAD_DIR, filename)
        storage.save(save_path)
        saved_index[filename] = save_path

    print(saved_index)
    qpath = saved_index.get("question.txt")
    if not qpath:
        print("here2")
        return jsonify({"error": "questions.txt missing after save"}), 400

    with open(qpath, "r", encoding="utf-8", errors="ignore") as f:
        questions_text = f.read()

    try:
        result_text = run_agent(questions_text, saved_index)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    try:
        cleaned = result_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        parsed = json.loads(cleaned.strip())

        if isinstance(parsed, dict):
            answers = list(parsed.values())
        elif isinstance(parsed, list):
            answers = parsed
        else:
            answers = [parsed]

        _preview("[Answers Preview]", jsonify(answers))
        return jsonify(answers), 200

    except Exception as e:
        print(result_text)
        logger.error("[api] Failed to parse JSON: %s", e)
        return Response(result_text, status=200, mimetype="text/plain; charset=utf-8")

if __name__ == "__main__":
    logger.info("Starting server on 0.0.0.0:%s (model=%s, upload_dir=%s, image_budget=%d)",
                os.environ.get("PORT", "8000"), OPENAI_MODEL, UPLOAD_DIR, MAX_IMAGE_BYTES)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
