import os
import json
import logging
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    try:
        # Lấy đường dẫn tuyệt đối của thư mục hiện tại
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.json')
        
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except FileNotFoundError:
        logger.error(f"config.json file not found at {config_path}")
        raise Exception(f"config.json file not found at {config_path}")
    except json.JSONDecodeError:
        logger.error("Error decoding config.json")
        raise Exception("Error decoding config.json")

def load_documents_from_folder(folder_path):
    if not os.path.exists(folder_path):
        logger.error(f"Folder path does not exist: {folder_path}")
        raise Exception(f"Folder path does not exist: {folder_path}")
        
    documents_pdf = []
    documents_doc = []
    
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if file.endswith('.pdf'):
                        loader_pdf = PyPDFLoader(file_path)
                        documents_pdf.extend(loader_pdf.load())
                        logger.info(f"Loaded PDF: {file_path}")
                    elif file.endswith('.docx'):
                        loader_doc = Docx2txtLoader(file_path)
                        documents_doc.extend(loader_doc.load())
                        logger.info(f"Loaded DOCX: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {str(e)}")
                    continue
    
        all_documents = documents_pdf + documents_doc
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    except Exception as e:
        logger.error(f"Error in load_documents_from_folder: {str(e)}")
        raise

def create_db(all_documents, api_key):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=150
        )
        chunks = text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(chunks)} chunks from documents")

        # Chỉ định đường dẫn tới thư mục có quyền ghi
        persist_directory = os.path.join(os.path.expanduser("~"), "DATN_test", "chroma_db")
        
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            # Set quyền 777 cho thư mục
            os.chmod(persist_directory, 0o777)
            logger.info(f"Created directory: {persist_directory}")

        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Khởi tạo Chroma với client_settings
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        
        logger.info("Database created successfully")
        return db
    except Exception as e:
        logger.error(f"Error in create_db: {str(e)}")
        raise

def main():
    try:
        # Load config và lấy API key
        config = load_config()
        api_key = config.get('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY not found in config.json")
            raise Exception("OPENAI_API_KEY not found in config.json")

        file_path = "/home/minhlahanhne/DATN_test/data"
        documents = load_documents_from_folder(file_path)
        db = create_db(documents, api_key)
        logger.info("Documents vectorized and saved to ChromaDB successfully")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()