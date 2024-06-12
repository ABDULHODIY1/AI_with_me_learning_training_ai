import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# JSONL faylini yuklash
data = []
with open("/DrAI/codingdatafpylarge.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# Dataset obyektini yaratish
dataset = Dataset.from_dict({
    "dialog": [item["text"] for item in data],
    "response": [item["label"] for item in data]
})

# Train va validation setlarni yaratish
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# Model va Tokenizer ni yuklash
model_name = "microsoft/DialoGPT-medium"
# model_name = "Path/toPath"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Padding tokenini o'rnatish
tokenizer.pad_token = tokenizer.eos_token

# Ma'lumotlar to'plamini tokenizatsiya qilish funksiyasi
def tokenize_function(examples):
    inputs = [dialog + tokenizer.eos_token + response for dialog, response in zip(examples['dialog'], examples['response'])]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Tokenizatsiya qilish
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["dialog", "response"])
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["dialog", "response"])

# Trening argumentlarini sozlash
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Kichik batch size
    per_device_eval_batch_size=4,  # Kichik batch size
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=4  # Effektiv batch size'ni oshirish
)


# Trainer ga start berish
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Modelni o'qitish
trainer.train()

# Model va tokenizer ni saqlash
model.save_pretrained("./results")
tokenizer.save_pretrained("./results")

print("Model va tokenizer saqlandi.")
