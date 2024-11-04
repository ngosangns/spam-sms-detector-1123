# Spam SMS & URL Detector

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

## Export installed packages versions

```sh
pip freeze > requirements.txt
```

## Download data

1. Download at: https://drive.google.com/file/d/1c0rxN8TnaBAhiMMBboaLZChoqaAwBMr8/view?usp=sharing
2. Locale the `data.zip` to the root of project.
3. Unzip: `unzip data.zip`
4. Zip data (if needed): `zip -r data.zip data`
