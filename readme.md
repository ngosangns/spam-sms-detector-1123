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
pip free > requirements.txt
```

## Download NLTK assets

```sh
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
python -c "import nltk; nltk.download()"
```

