# Nvidia-Hackathon
# CodeDoc Agent - AI-powered Code Analysis and Documentation Tool

**CodeDoc Agent** is a powerful AI-powered tool designed to help developers analyze, document, and refactor their codebases. Built using **NVIDIA Technologies** and various AI models, this tool automates code review tasks, generates docstrings, and answers questions related to code functionality, making it a must-have for improving developer productivity.

## Features

- **Code Analysis**: Answer questions like "What does this function do?" or "Where is logging handled?" by querying the codebase.
- **Auto Doc Generation**: Automatically generates function/class-level docstrings.
- **Refactor Suggestions**: Suggest improvements or optimizations to improve code quality or security.
- **Supports Python**: Works with Python codebases (zipped files).
- **Interactive UI**: Powered by Streamlit for easy interaction.

## Technologies Used

- **NVIDIA Triton Inference Server**: High-performance AI inference server to run models.
- **NVIDIA NeMo**: Used for embeddings and generating answers to code-related questions.
- **Streamlit**: Framework for building an interactive and user-friendly web interface.
- **Tree-sitter**: Parsing and analyzing Python code for extracting functions and code structure.
- **Ollama (for initial development)**: For processing natural language queries in the project. Replaced with NVIDIA models for this version.

## Installation

### Step 1: Install Dependencies

You will need **Python 3.8+** and **pip** to run this project.

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/codedoc-agent.git
    cd codedoc-agent
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Set Up NVIDIA Triton Inference Server

1. **Download and Set Up NVIDIA Triton Server** (For inference):
   Follow the official [NVIDIA Triton documentation](https://github.com/triton-inference-server/server) to set up Triton Inference Server locally. 

2. Ensure that your Triton server is up and running on `localhost:8000`.

### Step 3: Run the Application

1. To start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and navigate to `http://localhost:8501` to interact with the app.

### Step 4: Upload Python Code

Once the app is running:

- Upload a `.zip` file containing Python code for analysis.
- Select a Python file from the extracted zip.
- Ask questions, view auto-generated docstrings, or get refactor suggestions.

## Usage

### Example Workflow:

1. Upload a `.zip` file containing your Python code.
2. Select a Python file from the list.
3. Query the system to:
   - **Ask questions**: E.g., "What does this function do?" or "Where is logging handled?"
   - **Generate Docstrings**: Get docstrings for each function.
   - **Refactor Suggestions**: Receive suggestions to improve the code.

## Development

If you want to contribute to this project or run it locally, make sure you have the following tools installed:

- **Docker**: To run the Triton server in a container (optional but recommended).
- **Python 3.8+**: Ensure that you are using Python 3.8 or higher.
- **NVIDIA GPU**: If using Triton with GPU support, ensure that your system has an NVIDIA GPU with the necessary drivers and CUDA support.

### Testing

The project includes testing for major components. To run tests, use:

```bash
pytest
