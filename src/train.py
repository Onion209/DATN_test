
from sklearn.model_selection import train_test_split
from convert_data import load_data
from datasets import Dataset
from trl import SFTTrainer
from convert_data import load_model, load_tokenizer
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments, TextStreamer
max_seq_length = 2048

model, tokenizer = load_model()
dataset = load_data()
dataset_list = list(dataset)
train_data, eval_data = train_test_split(
    dataset_list,
    test_size=0.2,
    random_state=42
)
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# Khởi tạo trainer với validation set
trainer = SFTTrainer(
    model=model,
    tokenizer=load_tokenizer(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        num_train_epochs=20,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        weight_decay=0.05,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        optim="adamw_8bit",
        output_dir="output",
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    ),
)

# Bắt đầu training
trainer.train()