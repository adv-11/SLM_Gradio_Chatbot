# modules/document_processor.py
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS

def process_document(file_obj, api_key_valid):
    """
    Process an uploaded document and create a vector store.
    
    Args:
        file_obj: Uploaded file object
        api_key_valid: Boolean indicating if API key is valid
        
    Returns:
        tuple: (vector_store, status_message)
    """
    if not api_key_valid:
        return None, "Please set a valid API key before uploading documents. ⚠️"
    
    if file_obj is None:
        return None, ""
        
    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_obj.name)[1]) as temp_file:
            temp_file.write(file_obj.read())
            temp_file_path = temp_file.name
        
        # Select appropriate loader based on file type
        file_extension = os.path.splitext(file_obj.name)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
        else:  # Default to text loader
            loader = TextLoader(temp_file_path)
            
        # Load and split the document
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceHubEmbeddings(huggingfacehub_api_token=os.getenv('HF_TOKEN'))
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return vector_store, f"✅ Document processed successfully: {file_obj.name}"
        
    except Exception as e:
        return None, f"❌ Error processing document: {str(e)}"