import os
import glob
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
from imblearn.over_sampling import RandomOverSampler
from datasets import Dataset

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

def preprocess_data(sms_data, labels):
    dataset = Dataset.from_dict({'text': sms_data, 'label': labels})
    return dataset

def train_nlp_model(train_dataset, val_dataset):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir='./bert-results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained('./bert-results')
    tokenizer.save_pretrained('./bert-results')
    return model, tokenizer

def evaluate_model(model, tokenizer, test_dataset):
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt', device=0 if torch.cuda.is_available() else -1)
    predictions = pipeline(test_dataset['text'])
    y_pred = [pred['label'] == 'LABEL_1' for pred in predictions]
    y_true = test_dataset['label']
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    sms_directory = './sms-data'  # Replace with the actual directory

    sms_data, labels = load_sms_data(sms_directory)

    X_train_data, X_test_data, y_train, y_test = train_test_split(sms_data, labels, test_size=0.2, random_state=42)

    X_train_data, y_train = balance_dataset(X_train_data, y_train)

    train_dataset = preprocess_data(X_train_data, y_train)
    test_dataset = preprocess_data(X_test_data, y_test)

    # model, tokenizer = train_nlp_model(train_dataset, test_dataset)

    # Load the model and tokenizer from the saved directory
    model = BertForSequenceClassification.from_pretrained('./bert-results')
    tokenizer = BertTokenizer.from_pretrained('./bert-results')

    evaluate_model(model, tokenizer, test_dataset)