from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Load PDF data
# pdf_reader = SimpleDirectoryReader(input_files=[r"C:\Users\Pradeep\Desktop\Data\AI Test\Resume.pdf"])
# pdf_docs = pdf_reader.load_data()

def load_pdf(path):
    reader = SimpleDirectoryReader(input_files=[path])
    return reader.load_data()

# --- Create PDF Query Engine ---
def create_pdf_query_engine(pdf_path):
    pdf_docs = load_pdf(pdf_path)
    
    # Initialize embedding model and LLM
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    llm = Ollama(model="llama3", request_timeout=300.0, temperature=0)

    # Build vector index and query engine
    pdf_index = VectorStoreIndex.from_documents(pdf_docs, embed_model=embed_model)
    pdf_query_engine = pdf_index.as_query_engine(llm)
    
    return pdf_query_engine

# # Initialize models
# embed_model = OllamaEmbedding(model_name="nomic-embed-text")
# llm = Ollama(model="llama3", request_timeout=300.0, temperature=0)

# # Create PDF index and query engine
# pdf_index = VectorStoreIndex.from_documents(pdf_docs, embed_model=embed_model)
# pdf_query = pdf_index.as_query_engine(llm)



