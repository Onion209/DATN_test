import streamlit as st
from main import setup_qa_chain
import os
from datetime import datetime

# Khởi tạo session state cho conversations
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None

# Tạo sidebar
with st.sidebar:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSr2-hIfIOEB2-bok5hY83nxSQhmqOr0ANvTw&s",
            width=150,
            use_container_width=True
        )
    
    st.title("Model Configuration")
    
    # Chọn Embedding Model
    embedding_model = st.selectbox(
        "Choose Embedding Model",
        ["OpenAI Embeddings", "HuggingFace Embeddings"],
        index=0
    )
    
    # Chọn LLM Model
    llm_model = st.selectbox(
        "Choose LLM Model",
        ["GPT-4-mini", "Llama2", "Llama2-Finetuned"],
        index=0
    )
    
    # Nút để áp dụng cấu hình
    if st.button("Apply Configuration"):
        st.session_state.qa_chain = setup_qa_chain(
            embedding_model=embedding_model,
            llm_model=llm_model
        )
        st.success(f"Applied: {embedding_model} with {llm_model}")

    # Phần quản lý hội thoại
    st.markdown("---")  # Đường kẻ phân cách
    st.subheader("Conversations")
    
    # Nút tạo cuộc trò chuyện mới
    if st.button("🆕 New Chat"):
        new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.conversations[new_id] = {
            "title": f"Chat {len(st.session_state.conversations) + 1}",
            "messages": []
        }
        st.session_state.current_conversation_id = new_id
        st.session_state.chat_history = []
        st.rerun()

    # CSS cho khu vực cuộn
    st.markdown("""
        <style>
            [data-testid="stExpander"] div[data-testid="stVerticalBlock"] {
                max-height: 300px;
                overflow-y: auto;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sử dụng expander để tạo khu vực có thể cuộn
    with st.expander("Chat History", expanded=True):
        for conv_id, conv_data in st.session_state.conversations.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📝 {conv_data['title']}", key=f"conv_{conv_id}"):
                    st.session_state.current_conversation_id = conv_id
                    st.session_state.chat_history = conv_data['messages']
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{conv_id}"):
                    del st.session_state.conversations[conv_id]
                    if st.session_state.current_conversation_id == conv_id:
                        st.session_state.current_conversation_id = None
                        st.session_state.chat_history = []
                    st.rerun()

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 16px; font-style: italic;'>One love, one future.</p>", 
        unsafe_allow_html=True
    )

# Main content
st.title("💬 HUST Admissions Consulting Assistant")

# Hiển thị tiêu đề cuộc trò chuyện hiện tại
if st.session_state.current_conversation_id:
    current_chat = st.session_state.conversations[st.session_state.current_conversation_id]
    st.subheader(f"Current Chat: {current_chat['title']}")

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = setup_qa_chain()

# Hiển thị lịch sử chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.markdown("**Nguồn tham khảo:**")
            for source in message["sources"]:
                st.markdown(f"- {source}")

# Xử lý input từ người dùng
user_input = st.chat_input("Hãy đặt câu hỏi về tuyển sinh...")

if user_input and st.session_state.current_conversation_id:
    # Hiển thị câu hỏi của user
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Xử lý câu trả lời
    with st.chat_message("assistant"):
        result = st.session_state.qa_chain({"query": user_input})
        answer = result["result"]
        
        # Lọc và format sources
        sources = set()
        for doc in result["source_documents"]:
            source = doc.metadata.get('source', 'Unknown source')
            sources.add(os.path.basename(source))
        
        # Hiển thị câu trả lời và sources
        st.markdown(answer)
        if sources:
            st.markdown("**Nguồn tham khảo:**")
            for source in sources:
                st.markdown(f"- {source}")

        # Lưu vào chat history và cập nhật conversation
        message = {
            "role": "user",
            "content": user_input
        }
        st.session_state.chat_history.append(message)
        
        message = {
            "role": "assistant",
            "content": answer,
            "sources": list(sources)
        }
        st.session_state.chat_history.append(message)
        
        # Cập nhật conversation history
        st.session_state.conversations[st.session_state.current_conversation_id]["messages"] = st.session_state.chat_history