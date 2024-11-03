# SMS Spam Detector

- **Python Version:** 3.12.0
- **Node Package Manager:** Yarn (install globally with `npm i -g yarn`)

## Environment Setup

Create a virtual environment if it doesn’t already exist:

```sh
python3.12 -m venv venv
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

## Datasets

- https://openscience.vn/chi-tiet-du-lieu/bo-du-lieu-thu-thap-cac-binh-luan-youtube-tin-nhan-sms-tweet-de-phat-hien-spam--291
- https://www.kaggle.com/datasets/bwandowando/philippine-spam-sms-messages
- https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- https://www.kaggle.com/datasets/tapakah68/spam-text-messages-dataset
- https://www.kaggle.com/search?q=spam+sms+in%3Adatasets
- https://www.kaggle.com/datasets/rtatman/the-national-university-of-singapore-sms-corpus