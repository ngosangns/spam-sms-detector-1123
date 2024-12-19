# Spam SMS & URL Detector

- uv:

  - Windows:

```sh
winget install --id=astral-sh.uv -e
refreshenv # Command Promt
```

- python 3.12.7:

```sh
uv python install 3.12.7
uv python pin 3.12.7
```

- Taskfile:

  - Windows: `winget install Task.Task`

## Download NLTK assets

- Windows:

```sh
uv run python -c "import nltk; nltk.download()"
```

- MacOS / Linux:

```sh
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
uv run python -c "import nltk; nltk.download()"
```

## Comparing models

Open `./notebooks/sms_compare.ipynb` and `./notebooks/url_compare.ipynb` and run all commands.

## Running the Web Application

### Install Packages

To install the necessary packages, run the following commands: `task install`

### Run Application for Development

To start the application in development mode, use: `task dev`

### Run Application in Production Mode

To start the application in production mode, use: `task run`

To start the Telegram bot, use: `task tele` (bot username: `@sms_spam_detector_1123_bot`)

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

## URL

### Datasets

- https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector
