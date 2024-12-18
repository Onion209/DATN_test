import json
import pandas as pd
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
from datasets import load_dataset

# Biến global để cache tokenizer
_cached_tokenizer = None
max_seq_length = 2048

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    return model, tokenizer

def load_tokenizer(tokenizer):
    global _cached_tokenizer
    if _cached_tokenizer is None:
        _, base_tokenizer = tokenizer
        _cached_tokenizer = get_chat_template(
            base_tokenizer,
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
            chat_template="chatml",
        )
    return _cached_tokenizer

def convert_csv_to_chat_format(dataframe, output_file):
    if 'question' not in dataframe.columns or 'answer' not in dataframe.columns:
        raise ValueError("DataFrame must contain 'question' and 'answer' columns")
        
    formatted_data = []
    for _, row in dataframe.iterrows():
        if pd.isna(row['question']) or pd.isna(row['answer']):
            continue  # Skip rows with missing values
            
        conversation = [
            {"from": "human", "value": str(row['question']).strip()},
            {"from": "gpt", "value": str(row['answer']).strip()}
        ]
        formatted_data.append({"conversations": conversation})

    if not formatted_data:
        raise ValueError("No valid conversations were created")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

def apply_template(examples):
    messages = examples["conversations"]
    texts = []
    tokenizer = load_tokenizer()  # Load tokenizer một lần ở đầu hàm
    
    for message in messages:
        # Kiểm tra và xử lý các giá trị None
        if message is None:
            continue

        # Đảm bảo các tin nhắn trong conversation không có None
        valid_messages = []
        for msg in message:
            if msg is None or msg["value"] is None:
                continue
            valid_messages.append(msg)

        if valid_messages:  # Chỉ xử lý nếu có tin nhắn hợp lệ
            try:
                formatted_text = tokenizer.apply_chat_template(  # Sử dụng tokenizer đã load
                    valid_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(formatted_text)
            except Exception as e:
                print(f"Error processing message: {e}")
                texts.append("")  # Thêm chuỗi rỗng nếu xử lý thất bại
        else:
            texts.append("")  # Thêm chuỗi rỗng cho các trường hợp không có tin nhắn hợp lệ

    return {"text": texts}

def load_data():
    try:
        path = "/home/minhlahanhne/DATN_test/Q&A/Q&A_tuyensinh - Trang tính1.csv"
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV file is empty")
            
        convert_csv_to_chat_format(df, 'training_data.json')
        dataset = load_dataset("json", data_files={"train": "training_data.json"}, split="train")
        dataset = dataset.map(apply_template, batched=True)
        return dataset
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {path}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise