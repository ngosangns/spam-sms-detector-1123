from flask import Flask, request, jsonify
from models import BERTClassifier
from models import SVMClassifier
from models import NaiveBayesClassifier
from models import RandomForestClassifier
from models import LogisticRegressionClassifier
from models import KNNClassifier
from models import GradientBoostingClassifier

RESULT_DIR = './ml-models'

# Load the BERT model
model = SVMClassifier(RESULT_DIR)
model.load_model()

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_sms():
    data = request.get_json()
    sms = data.get('sms', '')

    if not sms:
        return jsonify({'error': 'No SMS provided'}), 400

    # Predict using the model
    prediction = model.predict(sms)
    sms_type = 'spam' if prediction[0] == 1 else 'ham'

    return jsonify({'type': sms_type})

if __name__ == '__main__':
    app.run(debug=True)