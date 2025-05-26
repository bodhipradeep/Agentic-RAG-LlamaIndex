import streamlit as st
from tools import create_tools
from retriever import create_pdf_query_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Streamlit app title and layout
st.set_page_config(page_title="Agentic RAG App", layout="wide")
st.title("Agentic RAG System")  

# Sidebar file upload
st.sidebar.header("📂 Upload Files")
pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history first
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Proceed only if both files are uploaded
if pdf_file and csv_file:
    # Save uploaded files temporarily
    pdf_path = f"temp_uploaded_pdf.pdf"
    csv_path = f"temp_uploaded_csv.csv"

    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())
    with open(csv_path, "wb") as f:
        f.write(csv_file.read())

    # Create tools and agent
    csv_tool, search_tool, code_tool, dataset_info_tool = create_tools(csv_path)
    pdf_query = create_pdf_query_engine(pdf_path)

    tools = [
        QueryEngineTool(
            query_engine=pdf_query,
            metadata=ToolMetadata(
                name="pdf_document_query",
                description="Query information from the loaded PDF document. Use this for questions about PDF content, document analysis, or extracting specific information from the PDF.",
            ),
        ),
        csv_tool,
        search_tool,
        code_tool,
        dataset_info_tool,
    ]

    # Load prompt
    with open("agent_prompt.txt", 'r', encoding='utf-8') as f:
        agent_prompt = f.read()

    # Initialize LLM agent
    llm = Groq(model="qwen-qwq-32b")
    # llm = Ollama(model='qwen3', temperature=0, request_timeout=300.0)
    agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True, system_prompt=agent_prompt)

    # Input box for user query
    user_query = st.chat_input("Enter your query")
    
    if user_query:
        # Add user message to chat history and display immediately
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.spinner("Thinking..."):
            try:
                response = agent.chat(user_query)
                # Get the response text properly
                if hasattr(response, 'response'):
                    response_text = str(response.response)
                else:
                    response_text = str(response)
                
                # Clean any residual thinking tags if they exist (as string)
                if "</think>" in response_text:
                    response_text = response_text.split("</think>")[-1]
                
                # Add assistant response to chat history and display
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)

else:
    st.info("Please upload both a PDF and a CSV file to begin.")