# Spam SMS & URL Detector

- python 3.12.7 - pyenv - anaconda3-2024.10-1

## Environment Setup

Create a virtual environment if it doesn’t already exist:

```sh
python -m venv venv
```

Activate the virtual environment:

```sh
source ./venv/bin/activate
```

## Install Dependencies

Install Python packages:

```sh
pip install -r requirements.txt
```

## Export installed packages versions

```sh
pip freeze > requirements.txt
```

## Download NLTK assets

```sh
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
python -c "import nltk; nltk.download()"
```

## Running the Web Application

### Production Mode

```sh
cd web
yarn
yarn build
cd ..
fastapi run app.py
```

### Development Mode

1. Start the front-end file watcher:

```sh
cd web
yarn
yarn watch
```

2. Start the back-end server with live reloading:

```sh
fastapi dev app.py
```

## SMS

### Datasets

- https://openscience.vn/chi-tiet-du-lieu/bo-du-lieu-thu-thap-cac-binh-luan-youtube-tin-nhan-sms-tweet-de-phat-hien-spam--291
- https://www.kaggle.com/datasets/bwandowando/philippine-spam-sms-messages
- https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- https://www.kaggle.com/datasets/tapakah68/spam-text-messages-dataset
- https://www.kaggle.com/search?q=spam+sms+in%3Adatasets
- https://www.kaggle.com/datasets/rtatman/the-national-university-of-singapore-sms-corpus
- https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

### Refs

- https://github.com/agupta98/SpamDetection
- https://github.com/kishan-1721/SMS-Spam-Detection
- https://github.com/ksdkamesh99/Spam-Classifier
- https://github.com/aniass/Spam-detection
- https://github.com/AHMEDSANA/Spam-and-Ham-text-classifier
- https://github.com/vaibhavbichave/Phishing-URL-Detection
- https://github.com/strikoder/SpamFilter
- https://keras.io/guides/training_with_built_in_methods
- https://docshare.tips/vietnamese-text-clasification_5765770db6d87fd2a78b4d82.html
- https://vnopenai.github.io/ai-doctor/nlp/vn-accent/n-grams
- https://github.com/sonlam1102/vispamdetection
