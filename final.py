import streamlit as st
import zipfile
import os
import shutil
import ast
from ollama import Client

# === CONFIG ===
client = Client(host='http://localhost:11434')  # Default Ollama host

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

def ask_llm(prompt, model="llama3"):  # You can change the model as needed
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error from Ollama: {e}"

def extract_functions_from_code(code):
    try:
        tree = ast.parse(code)
        return [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    except:
        return []

# === STREAMLIT UI ===
st.set_page_config(page_title="CodeDoc Agent MVP", layout="wide")
st.title("🔧 CodeDoc Agent MVP")
st.markdown("Upload a .zip file containing Python code. Then ask questions, generate docstrings, or get refactor tips.")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("Upload Codebase (.zip)", type=["zip"])

if uploaded_file:
    py_files = extract_zip(uploaded_file)
    if not py_files:
        st.warning("No .py files found in the uploaded ZIP.")
    else:
        selected_file = st.selectbox("Select a Python file to analyze", py_files)
        file_path = os.path.join("codebase", selected_file)
        code = load_code(file_path)
        st.subheader("📄 Code Preview")
        st.code(code, language="python")

        # === Ask Questions ===
        st.subheader("💬 Ask about this file")
        user_query = st.text_input("Enter your question (e.g., 'What does this function do?')")
        if user_query:
            prompt = f"You are an expert software engineer. Answer the following question about this code:\n\n{user_query}\n\nCode:\n{code}"
            answer = ask_llm(prompt)
            st.text_area("💡 Answer", value=answer, height=200)

        # === Generate Docstrings ===
        st.subheader("📄 Auto-Generated Docstrings")
        functions = extract_functions_from_code(code)

        if functions:
            for func in functions:
                func_code = ast.get_source_segment(code, func)
                st.markdown(f"### 🧠 Function: {func.name}")
                st.code(func_code, language="python")

                doc_prompt = f"Generate a Python docstring for this function:\n\n{func_code}"
                docstring = ask_llm(doc_prompt)
                st.text_area("📘 Docstring", value=docstring, height=150)

                # === Refactor Suggestions ===
                refactor_prompt = f"Suggest improvements or refactors for this function:\n\n{func_code}"
                suggestions = ask_llm(refactor_prompt)
                st.text_area("🛠️ Refactor Suggestions", value=suggestions, height=150)
        else:
            st.info("No top-level functions found in this file.")
