from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def query_chroma_db(query):
    # Tải lại cơ sở dữ liệu Chroma đã lưu và cung cấp hàm embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Khởi tạo Chroma với embeddings
    vectordb = Chroma(persist_directory="./chroma_langchain_db", embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    results = retriever.invoke(query)
    return results
def main():
    # Đặt câu truy vấn của bạn ở đây
    query = "Email của TS. Đào Huy Toàn là gì? "
    results = query_chroma_db(query)
    print("Results:")
    for result in results:
        print(result.page_content)

if __name__ == "__main__":
    main()
