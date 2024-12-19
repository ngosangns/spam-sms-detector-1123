from keras.src.layers.preprocessing.text_vectorization import TextVectorization
from keras.src.legacy.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class StopWords(metaclass=SingletonMeta):
    def __init__(self):
        self.words = stopwords.words("english")


class PortStemmer(metaclass=SingletonMeta):
    def __init__(self):
        self.stemmer = PorterStemmer()


class LanStemmer(metaclass=SingletonMeta):
    def __init__(self):
        self.stemmer = LancasterStemmer()


class Lemmatizer(metaclass=SingletonMeta):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()


class BertTokenizerFactory:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class CountVectorizerFactory:
    def __init__(self, ngram_range=(2, 6), max_features=3000):
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range, max_features=max_features
        )


class TfidfVectorizerFactory:
    def __init__(self, stop_words="english", max_features=3000):
        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words, max_features=max_features
        )


class TextVectorizationFactory:
    def __init__(self, output_mode="int", max_tokens=1000, output_sequence_length=1000):
        self.vectorization = TextVectorization(
            output_mode=output_mode,
            max_tokens=max_tokens,
            output_sequence_length=output_sequence_length,
        )


class KerasTokenizerFactory:
    def __init__(self, num_words=1000):
        self.tokenizer = Tokenizer(num_words=num_words)


if __name__ == "__main__":
    # Create instances using the singleton pattern
    port_stemmer = PortStemmer().stemmer
    lan_stemmer = LanStemmer().stemmer
    lemmatizer = Lemmatizer().lemmatizer

    count_vectorizer = CountVectorizerFactory().vectorizer
    bert_tokenizer = BertTokenizerFactory().tokenizer
    tfidf_vectorizer = TfidfVectorizerFactory().vectorizer
    text_vectorization = TextVectorizationFactory().vectorization
    keras_tokenizer = KerasTokenizerFactory().tokenizer

    # Example usage
    text = "This is an example text for preprocessing."

    # Using port_stemmer
    stemmed_text = port_stemmer.stem(text)
    print("Porter Stemmed Text:", stemmed_text)

    # Using lan_stemmer
    stemmed_text = lan_stemmer.stem(text)
    print("Lancaster Stemmed Text:", stemmed_text)

    # Using lemmatizer
    lemmatized_text = lemmatizer.lemmatize(text)
    print("Lemmatized Text:", lemmatized_text)

    # Using count_vectorizer
    vectorized_text = count_vectorizer.fit_transform([text])
    print("Count Vectorized Text:", vectorized_text.toarray())

    # Using bert_tokenizer
    tokenized_text = bert_tokenizer.encode(text, max_length=64, truncation=True)
    print("BERT Tokenized Text:", tokenized_text)

    # Using tfidf_vectorizer
    tfidf_text = tfidf_vectorizer.fit_transform([text])
    print("TF-IDF Vectorized Text:", tfidf_text.toarray())

    # Using text_vectorization
    vectorized_text = text_vectorization([text])
    print("Text Vectorization:", vectorized_text.numpy())

    # Using keras_tokenizer
    keras_tokenizer.fit_on_texts([text])
    tokenized_text = keras_tokenizer.texts_to_sequences([text])
    print("Keras Tokenized Text:", tokenized_text)
