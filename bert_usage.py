# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bodies = pd.read_csv("fnc-1-master/competition_test_bodies.csv")
stances = pd.read_csv("fnc-1-master/competition_test_stances_unlabeled.csv")

data_merged = stances.merge(bodies, on="Body ID")
data_merged['text'] = data_merged['Headline'] + " [SEP] " + data_merged['articleBody']

# incarcarea modelului si tokenizerului
model = BertForSequenceClassification.from_pretrained('./results')
tokenizer = BertTokenizer.from_pretrained('./results')
model.to(device)
model.eval()

label_map = {
    0: "unrelated",
    1: "discuss",
    2: "agree",
    3: "disagree"
}

predicted_labels = []

progress_bar = tqdm(data_merged['text'], desc="Procesare articole", unit="articol")
for text in progress_bar:
    encoded_inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=384,
        return_tensors="pt"
    )

    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    # obtinerea predictiei
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        predicted_labels.append(label_map[prediction])

data_merged['Stance'] = predicted_labels

output = data_merged[['Headline', 'Body ID', 'Stance']]

output.to_csv("bert_output.csv", index=False, encoding='utf-8')
print("Rezultatele au fost salvate în 'bert_output.csv'")