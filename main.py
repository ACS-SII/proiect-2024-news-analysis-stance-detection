import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Verificati daca GPU este disponibil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bodies = pd.read_csv("fnc-1-master/train_bodies.csv")
stances = pd.read_csv("fnc-1-master/train_stances.csv")

print(stances['Stance'].value_counts())

data_merged = stances.merge(bodies, on="Body ID").drop("Body ID", axis=1)
print(data_merged.columns)

# Concatenarea titlului si a corpului articolului
data_merged['text'] = data_merged['Headline'] + " [SEP] " + data_merged['articleBody']

# Codificarea etichetelor
label_encoder = LabelEncoder()
data_merged['label'] = label_encoder.fit_transform(data_merged['Stance'])

# Impartirea setului de date
train_df, val_df = train_test_split(data_merged[['text', 'label']], test_size=0.1, random_state=42)

# Conversia la formatul Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Incarcarea tokenizer-ului BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Functie de tokenizare
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)


# Aplicarea tokenizarii
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Specificarea coloanelor de intrare si eticheta
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Incarcarea modelului BERT pre-antrenat
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model.to(device)

def compute_metrics(pred):
    predictions, labels = pred
    predictions = predictions.argmax(axis=-1)  # Clasele cu probabilitatea maxima
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="no",
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics  # Tokenizer-ul nu mai este necesar
)

print(trainer.args.device)

# Antrenarea modelului
trainer.train()

# Evaluarea modelului
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
# another push