import os
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    # TFBertModel,
)
import torch
from transformers import (
    TextClassificationPipeline,
    TrainingArguments,
    Trainer,
)

from models.di import BertTokenizerFactory, SingletonMeta
from models.sms_ml_classifier import SMSClassifier


class SMSBERTClassifier(SMSClassifier):
    def __init__(self, model_dir):
        super().__init__("bert", model_dir)
        self.model: BertForSequenceClassification = None

        self.tokenizer: BertTokenizer = None
        self.tokenizer_path = os.path.join(
            self.model_dir, f"sms-{self.model_name}-tokenizer.pkl"
        )

    def train(self, X: np.ndarray, Y: np.ndarray):
        X1, X2, Y1, Y2 = train_test_split(
            X.copy(), Y.copy(), test_size=0.2, random_state=42
        )

        X1_df = Dataset.from_dict({"text": X1, "label": Y1})
        X2_df = Dataset.from_dict({"text": X2, "label": Y2})

        # tokenize
        if self.tokenizer is None:
            self.tokenizer = BertTokenizerFactory().tokenizer

        def tokenize_function(example):
            return self.tokenizer(
                example["text"], padding="max_length", truncation=True
            )

        X1_bert_tokenized = X1_df.map(tokenize_function, batched=True)
        X2_bert_tokenized = X2_df.map(tokenize_function, batched=True)

        if self.model is None:
            self.model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2
            )

        training_args = TrainingArguments(
            output_dir=self.model_path,
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=X1_bert_tokenized,
            eval_dataset=X2_bert_tokenized,
        )

        trainer.train()

    def predict(self, X: np.ndarray) -> np.ndarray[int]:
        if self.tokenizer is None:
            self.tokenizer = BertTokenizerFactory().tokenizer

        pipeline = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt",
            device=0 if torch.cuda.is_available() else -1,
        )
        Y_pred = pipeline(X)
        Y_pred = np.array([y["label"] == "LABEL_1" for y in Y_pred])
        return Y_pred

    def save(self):
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.tokenizer_path)

    def load(self):
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)


class SingletonSMSBERTClassifier(SMSBERTClassifier, metaclass=SingletonMeta):
    pass


# # Example of using BERT with DNN, but accuracy is not good, so I use BERT2 instead
# # BERT classifier
# class SMSBERTClassifier(SMSClassifier):
#     def __init__(self, model_dir):
#         super().__init__("bert", model_dir)
#         self.model: Model = None
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#     def build_bert_model(self, max_length):
#         input_ids = tf.keras.Input(shape=(max_length,), dtype="int32")
#         attention_masks = tf.keras.Input(shape=(max_length,), dtype="int32")

#         bert_model = TFBertModel.from_pretrained("bert-base-uncased")
#         output = bert_model([input_ids, attention_masks])
#         output = output[1]
#         output = tf.keras.layers.Dense(32, activation="relu")(output)
#         output = tf.keras.layers.Dropout(0.2)(output)
#         output = tf.keras.layers.Dense(1, activation="sigmoid")(output)

#         model = tf.keras.models.Model(
#             inputs=[input_ids, attention_masks], outputs=output
#         )
#         model.compile(
#             tf.keras.optimizers.Adam(lr=1e-5),
#             loss="binary_crossentropy",
#             metrics=["accuracy"],
#         )

#         self.model = model

#     def tokenize(self, X, max_length=64):
#         input_ids = []
#         attention_masks = []

#         for x in X:
#             encoded = self.tokenizer.encode_plus(
#                 x,
#                 add_special_tokens=True,
#                 max_length=max_length,
#                 padding="max_length",
#                 truncation=True,
#                 return_attention_mask=True,
#             )
#             input_ids.append(encoded["input_ids"])
#             attention_masks.append(encoded["attention_mask"])

#         return np.array(input_ids), np.array(attention_masks)

#     def train(
#         self,
#         X: np.ndarray,
#         Y: np.ndarray,
#         validation_split=0.2,
#         epochs=3,
#         verbose=1,
#         batch_size=10,
#     ):
#         # vectorize
#         bert_tokenize_max_length = 64
#         X_input_ids, X_attention_masks = self.tokenize(X.copy(), 64)

#         if self.model is None:
#             self.build_bert_model(bert_tokenize_max_length)

#         self.model.fit(
#             [X_input_ids, X_attention_masks],
#             Y,
#             validation_split=validation_split,
#             epochs=epochs,
#             verbose=verbose,
#             batch_size=batch_size,
#         )

#     def predict(self, X) -> np.ndarray[int]:
#         pipeline = TextClassificationPipeline(
#             model=self.model,
#             tokenizer=self.tokenizer,
#             framework="pt",
#             device=0 if torch.cuda.is_available() else -1,
#         )
#         Y_pred = pipeline(X)
#         Y_pred = np.array([y["label"] == "LABEL_1" for y in Y_pred])
#         return Y_pred

#     def save(self):
#         with open(self.model_path, "wb") as f:
#             pickle.dump(self.model, f)

#     def load(self):
#         with open(self.model_path, "rb") as f:
#             self.model = pickle.load(f)
