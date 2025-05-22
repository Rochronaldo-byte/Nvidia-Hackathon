import streamlit as st
import zipfile
import os
import shutil
import ast
import requests

# === CONFIG ===
NIM_URL = "http://localhost:8000/v1/chat/completions" 
NIM_MODEL = "meta/llama3"  

# === FUNCTIONS ===

def extract_zip(zip_file, extract_to="codebase"):
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    return [f for f in os.listdir(extract_to) if f.endswith(".py")]

def load_code(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def ask_llm(prompt, model=NIM_MODEL):
    try:
        response = requests.post(
            NIM_URL,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error contacting NVIDIA NIM: {e}"

def extract_functions_from_code(code):
    try:
        tree = ast.parse(code)
        return [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    except:
        return []

# === STREAMLIT UI ===
st.set_page_config(page_title="CodeDoc Agent", layout="wide")
st.title("🔧 CodeDoc Agent")
st.markdown("Upload a `.zip` file of Python files. Ask questions, get docstrings, and see refactor suggestions powered by **NVIDIA NIM**.")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📁 Upload Codebase (.zip)", type=["zip"])

if uploaded_file:
    py_files = extract_zip(uploaded_file)
    if not py_files:
        st.warning("No Python files found.")
    else:
        selected_file = st.selectbox("📄 Select a Python file", py_files)
        file_path = os.path.join("codebase", selected_file)
        code = load_code(file_path)
        st.subheader("🧾 Code Preview")
        st.code(code, language="python")

        # === Ask Question ===
        st.subheader("💬 Ask About This File")
        user_query = st.text_input("What would you like to know?")
        if user_query:
            full_prompt = f"You are an expert software engineer. Please answer the following question based on the code:\n\n{user_query}\n\nCode:\n{code}"
            response = ask_llm(full_prompt)
            st.text_area("💡 Answer from NVIDIA LLM", value=response, height=200)

        # === Docstrings + Refactor
        st.subheader("📄 Auto-Generated Docstrings + Suggestions")
        functions = extract_functions_from_code(code)

        if functions:
            for func in functions:
                func_code = ast.get_source_segment(code, func)
                st.markdown(f"### 🔹 Function: `{func.name}`")
                st.code(func_code, language="python")

                # Generate Docstring
                doc_prompt = f"Generate a Python docstring for this function:\n\n{func_code}"
                docstring = ask_llm(doc_prompt)
                st.text_area("📘 Docstring", value=docstring, height=150)

                # Refactor Suggestion
                refactor_prompt = f"Suggest improvements or refactors for this function:\n\n{func_code}"
                suggestions = ask_llm(refactor_prompt)
                st.text_area("🛠️ Refactor Suggestions", value=suggestions, height=150)
        else:
            st.info("No top-level functions found in this file.")
