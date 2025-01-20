import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from tqdm import tqdm
import sys

sys.stdout.reconfigure(encoding='utf-8')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bodies = pd.read_csv("fnc-1-master/train_bodies.csv")
stances = pd.read_csv("fnc-1-master/train_stances.csv")
data_merged = stances.merge(bodies, on="Body ID")
data_merged['text'] = data_merged['Headline'] + " [SEP] " + data_merged['articleBody']

label_map = {
    0: "unrelated",
    1: "discuss",
    2: "agree",
    3: "disagree"
}
label_to_num = {v: k for k, v in label_map.items()}
data_merged['label'] = data_merged['Stance'].map(label_to_num)

train_df, val_df = train_test_split(data_merged[['text', 'label']], test_size=0.1, random_state=42)

# conversia in dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=384
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# specificarea coloanelor
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Dataloader pentru antrenare și validare
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# incarcarea modelului si functiei de pierdere ponderate
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model.to(device)

# calcularea ponderilor claselor
class_counts = train_df['label'].value_counts().sort_index()
total_samples = len(train_df)
class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# definirea functiei de pierdere ponderata
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

# optimizarea
optimizer = AdamW(model.parameters(), lr=3e-5)

def compute_metrics(predictions, labels):
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    predictions = predictions.argmax(axis=-1) if predictions.ndim > 1 else predictions
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return accuracy, precision, recall, f1


epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoca {epoch + 1} - Antrenare", unit="batch")  # Progres antrenare
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).long()  # Convertim etichetele la tipul long

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)  # Calculăm pierderea folosind ponderile corectate
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())  # Afișăm pierderea curentă

    print(f"Epoca {epoch + 1}, Pierderea medie: {total_loss / len(train_loader)}")

    model.eval()
    val_labels = []
    val_predictions = []
    progress_bar = tqdm(val_loader, desc=f"Epoca {epoch + 1} - Validare", unit="batch")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            #colectare predictie si etichete
            pred_classes = torch.argmax(logits, dim=-1).cpu().numpy()
            val_predictions.extend(pred_classes)
            val_labels.extend(labels.cpu().numpy())

    # linste in array-uri numpy
    val_predictions = np.array(val_predictions).squeeze()
    val_labels = np.array(val_labels).squeeze()

    # Verificăm dimensiunile și valorile
    print(f"Tip val_predictions: {type(val_predictions)}, Tip val_labels: {type(val_labels)}")
    print(f"Dimensiuni val_predictions: {val_predictions.shape}, Dimensiuni val_labels: {val_labels.shape}")
    print(f"Exemple val_predictions: {val_predictions[:5]}, Exemple val_labels: {val_labels[:5]}")

    val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(
        predictions=val_predictions,
        labels=val_labels
    )

    print(
        f"Epoca {epoch + 1}, Acuratețe: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

model.save_pretrained('./results')
tokenizer.save_pretrained('./results')