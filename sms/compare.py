import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from models import SVMClassifier
from models import NaiveBayesClassifier
from models import RandomForestClassifier
from models import LogisticRegressionClassifier
from models import KNNClassifier
from models import GradientBoostingClassifier
from models import BERTClassifier

if __name__ == "__main__":
    SMS_DIR = '../data/sms-data'
    RESULT_DIR = './ml-models'
    IS_TRAINING = True
    
    classifiers = [
        SVMClassifier(RESULT_DIR),
        NaiveBayesClassifier(RESULT_DIR),
        RandomForestClassifier(RESULT_DIR),
        LogisticRegressionClassifier(RESULT_DIR),
        KNNClassifier(RESULT_DIR),
        GradientBoostingClassifier(RESULT_DIR),
        # BERTClassifier(RESULT_DIR),
    ]
    model_names = list([classifier.model_name for classifier in classifiers])
    accuracies = []

    for classifier in classifiers:
        sms_data, labels = classifier.load_sms_data(SMS_DIR)
        X_train, y_train, X_test, y_test = classifier.preprocess_data(sms_data, labels)

        if IS_TRAINING:
            if classifier.model_name == 'bert-base-uncased':
                classifier.train_model(X_train, X_test)
            else:
                classifier.train_model(X_train, y_train)
            classifier.save_model()
        else:
            classifier.load_model()
        
        print(f'Evaluating {classifier.model_name}...')
        y_test, y_pred = classifier.evaluate_model(X_test, y_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        accuracies.append(accuracy * 100)

    plt.figure(figsize=(12, 6))  # Set default window width to 1000 pixels (10 inches)
    plt.bar(model_names, accuracies)
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Comparison')

    # Adding accuracy labels on top of each bar
    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 1, f'{accuracy:.2f}%', ha='center', va='bottom')

    plt.show()
