import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from models.url_ml_catboost_classifier import SingletonURLMLCatBoostClassifier
from models.url_ml_desicion_tree_classifier import SingletonURLMLDecisionTreeClassifier
from models.url_ml_gradient_boosting_classifier import (
    SingletonURLMLGradientBoostingClassifier,
)
from models.url_ml_knn_classifier import SingletonURLMLKNNClassifier
from models.url_ml_logistic_regression_classifier import (
    SingletonURLMLLogisticRegressionClassifier,
)
from models.url_ml_mlp_classifier import SingletonURLMLMLPClassifier
from models.url_ml_naive_bayes_classifier import SingletonURLMLNaiveBayesClassifier
from models.url_ml_random_forest import SingletonURLMLRandomForestClassifier
from models.url_ml_svm_classifier import SingletonURLMLSVMClassifier
from utils.url_utils import load_data_from_csv


def evaluate_classifiers(classifiers, X_train, Y_train, X_test, Y_test, is_training):
    accuracies = []
    for classifier in classifiers:
        if is_training:
            classifier.train(X_train.copy(), Y_train.copy())
            classifier.save()
        else:
            classifier.load()

        print(f"Evaluating {classifier.model_name}...")
        Y_pred = classifier.predict(X_test.copy())
        accuracy = accuracy_score(Y_test.copy(), Y_pred)
        print(f"Accuracy: {accuracy}")
        accuracies.append(accuracy * 100)
    return accuracies


def plot_accuracies(model_names, accuracies):
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, accuracies)
    plt.xlabel("Model")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Comparison")
    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 1, f"{accuracy:.2f}%", ha="center", va="bottom")
    plt.show()


if __name__ == "__main__":
    URL_CSV_PATH = "./data/url-data.csv"
    MODEL_DIR = "./trained_models"
    IS_TRAINING = True

    classifiers = [
        SingletonURLMLSVMClassifier(MODEL_DIR),
        SingletonURLMLNaiveBayesClassifier(MODEL_DIR),
        SingletonURLMLRandomForestClassifier(MODEL_DIR),
        SingletonURLMLLogisticRegressionClassifier(MODEL_DIR),
        SingletonURLMLKNNClassifier(MODEL_DIR),
        SingletonURLMLGradientBoostingClassifier(MODEL_DIR),
        SingletonURLMLCatBoostClassifier(MODEL_DIR),
        SingletonURLMLDecisionTreeClassifier(MODEL_DIR),
        SingletonURLMLMLPClassifier(MODEL_DIR),
    ]
    model_names = [classifier.model_name for classifier in classifiers]

    X, Y = load_data_from_csv(URL_CSV_PATH)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    accuracies = evaluate_classifiers(
        classifiers, X_train, Y_train, X_test, Y_test, IS_TRAINING
    )
    plot_accuracies(model_names, accuracies)
