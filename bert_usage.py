# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm  # Bara de progres

# Verificarea dispozitivului (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Încărcarea datelor
bodies = pd.read_csv("fnc-1-master/competition_test_bodies.csv")
stances = pd.read_csv("fnc-1-master/competition_test_stances_unlabeled.csv")

# Combinarea datelor
data_merged = stances.merge(bodies, on="Body ID")
data_merged['text'] = data_merged['Headline'] + " [SEP] " + data_merged['articleBody']

# Încarcarea modelului și tokenizer-ului
model = BertForSequenceClassification.from_pretrained('./results')
tokenizer = BertTokenizer.from_pretrained('./results')
model.to(device)
model.eval()  # Mod evaluare

# Maparea etichetelor numerice la etichete textuale
label_map = {
    0: "unrelated",
    1: "discuss",
    2: "agree",
    3: "disagree"
}

# Procesare și obținerea predicțiilor
predicted_labels = []

# Tokenizarea și predicțiile
progress_bar = tqdm(data_merged['text'], desc="Procesare articole", unit="articol")
for text in progress_bar:
    # Tokenizare
    encoded_inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=384,  # Adaptați dacă e necesar
        return_tensors="pt"
    )

    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    # Obținerea predicției
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        predicted_labels.append(label_map[prediction])  # Mapare la etichete textuale

# Adăugarea predicțiilor în DataFrame
data_merged['Stance'] = predicted_labels

# Selectarea coloanelor relevante
output = data_merged[['Headline', 'Body ID', 'Stance']]

# Salvarea rezultatelor
output.to_csv("bert_output.csv", index=False, encoding='utf-8')
print("Rezultatele au fost salvate în 'bert_output.csv'")
