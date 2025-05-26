import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import contextlib
from llama_index.core.tools import FunctionTool
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.llms.ollama import Ollama
import streamlit as st
import os


# --- CSV Loader ---
def load_csv(csv_path: str):
    return pd.read_csv(csv_path, encoding="cp1252")

# --- Create All Tools Based on CSV Path ---
def create_tools(csv_path: str):
    df = load_csv(csv_path)

    # Initialize LLM
    llm = Ollama(model="llama3", request_timeout=300.0, temperature=0)

    # CSV Query Engine
    csv_query_engine = PandasQueryEngine(df=df, llm=llm, verbose=True)

    def query_csv_tool(input: str):
        try:
            result = csv_query_engine.query(input)
            return str(result)
        except Exception as e:
            return f"Error querying CSV: {str(e)}"

    csv_tool = FunctionTool.from_defaults(
        fn=query_csv_tool,
        name="csv_data_query",
        description=(
            f"Use this tool to query the CSV dataset. "
            f"Available columns: {list(df.columns)}. "
            f"Dataset shape: {df.shape}. "
            f"Use this for data analysis, statistics, filtering, and getting insights from the data."
        )
    )

    # Load your Tavily API key from env
    tavily_tool = TavilyToolSpec(api_key=os.getenv("TAVILY_API_KEY"))

    # Get the underlying function from TavilyToolSpec
    tavily_function = tavily_tool.to_tool_list()[0].fn  # Only one tool in the list

    # Wrap it with FunctionTool for use in ReActAgent
    search_tool = FunctionTool.from_defaults(
        fn=tavily_function,
        name="web_search",
        description="Use this tool to search for current information on the web using Tavily, which provides real-time search results."
    )


    def execute_python_code(code: str):
        try:
            output_buffer = io.StringIO()
            exec_globals = {
                'plt': plt,
                'sns': sns,
                'pd': pd,
                'df': df,
                'print': lambda *args, **kwargs: print(*args, file=output_buffer, **kwargs)
            }
            with contextlib.redirect_stdout(output_buffer):
                exec(code, exec_globals)
            output = output_buffer.getvalue()

            # Capture all existing figures
            figures = []
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                figures.append(fig)

            # Display all figures in Streamlit
            if figures:
                for fig in figures:
                    st.pyplot(fig)
                    plt.close(fig)
                output += "\n[Plots displayed successfully in Streamlit]"

            return f"Code executed successfully.\nOutput:\n{output}" if output else "Code executed successfully."
        except Exception as e:
            return f"Code execution error: {str(e)}"

    code_tool = FunctionTool.from_defaults(
        fn=execute_python_code,
        name="code_executor",
        description=(
            "Executes Python code for analysis. You have access to: "
            "- pandas (as pd), matplotlib (as plt), seaborn (as sns) "
            "- The dataframe variable 'df' (already loaded, DON'T use pd.read_csv()) "
            "- Always show values on chart, Example -: ax.bar_label(ax.containers[0]) for bar/column chart"
            "Example usage: "
            "```python\n"
            "fig, ax = plt.subplots(figsize=(10,8))\n"
            "sns.countplot(x='trip_type', data=df)\n"
            "ax.bar_label(ax.containers[0])\n" # Shows value labels on bars
            "plt.title('Trip Type Distribution')\n"
            "```"
        )
    )

    def get_dataset_info():
        info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict('records')
        }
        return str(info)

    dataset_info_tool = FunctionTool.from_defaults(
        fn=get_dataset_info,
        name="dataset_info",
        description="Get detailed information about the loaded CSV dataset including columns, data types, shape, and sample data."
    )

    return csv_tool, search_tool, code_tool, dataset_info_tool