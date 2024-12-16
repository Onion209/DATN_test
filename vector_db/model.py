from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
import os
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Cấu hình logging chỉ cho __main__
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.json')
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def load_model(api_key):
    model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=api_key
    )
    return model

def create_prompt():
    template = """Bạn là giáo viên phòng tư vấn tuyển sinh của trường Đại học Bách Khoa Hà Nội.
    Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và ngắn gọn.
    Nếu không có thông tin để trả lời, hãy nói "Tôi không có đủ thông tin để trả lời câu hỏi này."

    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

def load_vector_db(api_key):
    try:
        persist_directory = os.path.join(os.path.expanduser("~"), "DATN_test", "chroma_db")
        embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        
        if not os.path.exists(persist_directory):
            raise Exception("Vector database not found. Please run vectorize_doc.py first.")
            
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        return db
    except Exception as e:
        logger.error(f"Error loading vector database: {str(e)}")
        raise

def create_qa_chain(db, model, prompt):
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}")
        raise

def get_answer(question):
    try:
        # Load config và API key
        config = load_config()
        api_key = config.get('OPENAI_API_KEY')
        
        if not api_key:
            raise Exception("OPENAI_API_KEY not found in config.json")
        
        # Load model, database và tạo chain
        model = load_model(api_key)
        db = load_vector_db(api_key)
        prompt = create_prompt()
        qa_chain = create_qa_chain(db, model, prompt)
        
        # Generate câu trả lời
        result = qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
        
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        raise

def main():
    try:
        # Test với một số câu hỏi mẫu
        questions = [
            "Điểm chuẩn ngành Toán tin năm 2024 là bao nhiêu?"
        ]
        
        for question in questions:
            print("\nCâu hỏi:", question)
            result = get_answer(question)
            print("\nCâu trả lời:", result["answer"])
            print("\nNguồn tham khảo:")
            # Lọc và in ra source documents không trùng lặp
            sources = set()
            for doc in result["source_documents"]:
                source = doc.metadata.get('source', 'Unknown source')
                if source not in sources:
                    sources.add(source)
                    print(f"- {os.path.basename(source)}")
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()