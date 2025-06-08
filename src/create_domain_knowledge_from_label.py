import json

label_path = "/root/ocr-repair/data/test/labels/5350224/1996/5350224-1996-0001-0037.json"
domain_knowledge_path = "/root/ocr-repair/data/domain_knowledge_test.json"

with open(label_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract and concatenate all annotation.text fields in order
annotations = data.get('annotations', [])
texts = [ann['annotation.text'] for ann in annotations if 'annotation.text' in ann]
concatenated_text = ' '.join(texts)

# Create domain knowledge file
output = {concatenated_text: {}}

with open(domain_knowledge_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Domain knowledge file created: {domain_knowledge_path}") 