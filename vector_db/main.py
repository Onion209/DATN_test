import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
import warnings
import streamlit as st
from langchain_community.llms import LlamaCpp
warnings.filterwarnings('ignore')

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.json')
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        raise

def create_prompt():
    template = """Bạn là giáo viên phòng tư vấn tuyển sinh của trường Đại học Bách Khoa Hà Nội.
    Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và ngắn gọn.
    Nếu không có thông tin để trả lời, hãy nói "Tôi không có đủ thông tin để trả lời câu hỏi này."

    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

def setup_qa_chain(embedding_model_choice="OpenAI", llm_model_choice="GPT-4-mini"):
    try:
        # Load config và API key
        config = load_config()
        api_key = config.get('OPENAI_API_KEY')
        
        if not api_key:
            raise Exception("OPENAI_API_KEY not found in config.json")

        # Khởi tạo embedding model dựa trên lựa chọn
        if embedding_model_choice == "OpenAI":
            embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        else:
            # Thêm các embedding model khác ở đây
            raise Exception("Embedding model not implemented yet")

        # Khởi tạo LLM model dựa trên lựa chọn
        if llm_model_choice == "GPT-4-mini":
            model = ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.5,
                openai_api_key=api_key
            )
        elif llm_model_choice == "Llama2":
            # Cấu hình cho Llama2
            model = LlamaCpp(
                model_path="path/to/llama2/model",
                temperature=0.5,
                max_tokens=2000
            )
        elif llm_model_choice == "Llama2-Finetuned":
            # Cấu hình cho Llama2 đã fine-tuned
            model = LlamaCpp(
                model_path="path/to/finetuned/llama2/model",
                temperature=0.5,
                max_tokens=2000
            )
        else:
            raise Exception("LLM model not supported")

        # Load vector database
        persist_directory = os.path.join(os.path.expanduser("~"), "DATN_test", "chroma_db")
        
        if not os.path.exists(persist_directory):
            raise Exception("Vector database not found. Please run vectorize_doc.py first.")
            
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )

        # Tạo QA chain
        prompt = create_prompt()
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain

    except Exception as e:
        st.error(f"Error setting up QA chain: {str(e)}")
        raise

