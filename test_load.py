import os
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader


# Đường dẫn đến thư mục cha chứa các thư mục con
data_path = "/home/minhlahanhne/DATN_test/data"

# Khởi tạo các loader cho file PDF và DOCX
loader_pdf = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
loader_doc = DirectoryLoader(data_path, glob="*.docx", loader_cls=Docx2txtLoader)

# Duyệt qua tất cả thư mục con trong thư mục data_path
documents_pdf = []
documents_doc = []

for root, dirs, files in os.walk(data_path):
    # Lọc các file PDF và DOCX trong từng thư mục con
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith('.pdf'):
            loader_pdf = PyPDFLoader(file_path)
            documents_pdf.extend(loader_pdf.load())
        elif file.endswith('.docx'):
            loader_doc = Docx2txtLoader(file_path)
            documents_doc.extend(loader_doc.load())

# Kết hợp tất cả tài liệu PDF và DOCX
all_documents = documents_pdf + documents_doc
print(all_documents)