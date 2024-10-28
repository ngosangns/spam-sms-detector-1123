import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from transformers import BertTokenizer
from datasets import Dataset as HFDataset

# Define an LSTM-based model for sequence classification
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out

def balance_dataset(X, y):
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample([[x] for x in X], y)
    X_resampled = [x[0] for x in X_resampled]  # Flatten the resampled X back to 1D
    return X_resampled, y_resampled

def load_sms_data(directory):
    sms_data = []
    labels = []
    for file_path in glob.glob(os.path.join(directory, '*.txt')):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            lines = content.split('\n')
            if len(lines) >= 2:
                label = lines[0].lower()
                sms_content = '\n'.join(lines[1:])
                sms_data.append(sms_content)
                labels.append(1 if label == 'spam' else 0)
    return sms_data, labels

def preprocess_data(sms_data, labels, tokenizer, max_len=128):
    tokenized_data = [tokenizer.encode(text, truncation=True, padding='max_length', max_length=max_len) for text in sms_data]
    dataset = HFDataset.from_dict({'text': tokenized_data, 'label': labels})
    return dataset

class SMSDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_nlp_model(train_dataset, val_dataset, vocab_size):
    embed_dim = 128
    hidden_dim = 64
    output_dim = 2
    model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    for epoch in range(3):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch['input_ids'], batch['labels']
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model

def evaluate_model(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=8)
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['input_ids'], batch['labels']
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())
            y_true.extend(labels.tolist())
    
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    sms_directory = './sms-data'  # Replace with the actual directory

    # Load and balance dataset
    sms_data, labels = load_sms_data(sms_directory)
    X_train_data, X_test_data, y_train, y_test = train_test_split(sms_data, labels, test_size=0.2, random_state=42)
    X_train_data, y_train = balance_dataset(X_train_data, y_train)

    # Initialize tokenizer and preprocess data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = preprocess_data(X_train_data, y_train, tokenizer)
    test_dataset = preprocess_data(X_test_data, y_test, tokenizer)

    # Convert Hugging Face Datasets to PyTorch-compatible Datasets
    train_dataset = SMSDataset({'input_ids': train_dataset['text']}, train_dataset['label'])
    test_dataset = SMSDataset({'input_ids': test_dataset['text']}, test_dataset['label'])

    # Train and evaluate the LSTM model
    vocab_size = tokenizer.vocab_size
    model = train_nlp_model(train_dataset, test_dataset, vocab_size)
    evaluate_model(model, test_dataset)