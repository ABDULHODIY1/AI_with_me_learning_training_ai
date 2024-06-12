import json
from datasets import load_dataset

# 1. Datasetni yuklab olish
dataset = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train')

# 2. JSON formatiga o'tkazish
transformed_data = []
for example in dataset:
    text_label = {
        'text': example['instruction'],
        'label': example['output']
    }
    transformed_data.append(text_label)

# 3. JSON faylni saqlash
output_file = 'transformed_dataset.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(transformed_data, f, ensure_ascii=False, indent=4)

print(f"Dataset JSON formatida saqlandi: {output_file}")
