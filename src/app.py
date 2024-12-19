import os

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from src.app_dto import Response, SMSRequest, URLRequest
from src.app_models import init_app_models
from src.models.url_feature_extractor import URLFeatureExtractor
from utils.sms_utils import (
    detect_email_addresses,
    detect_general_phone_number,
    detect_urls,
)
from utils.url_utils import extract_domain_path


# Function to load blacklist data from a CSV file
def load_blacklist_from_csv(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Blacklist file not found at {csv_path}")


# Load blacklists for emails, phone numbers, and URLs
sms_email_blacklist_path = load_blacklist_from_csv("./data/sms-email-blacklist.csv")
sms_phone_blacklist_path = load_blacklist_from_csv("./data/sms-phone-blacklist.csv")
url_blacklist_path = load_blacklist_from_csv("./data/url-blacklist.csv")

# Initialize the application models
app_models = init_app_models()

# Create a FastAPI app instance
app = FastAPI()


# Endpoint to classify SMS messages
@app.post("/sms/classify")
async def classify_sms(request: SMSRequest):
    sms = request.sms
    if not sms:
        raise HTTPException(status_code=400, detail="No SMS provided")

    # Check for blacklisted email addresses in the SMS
    emails_iter = detect_email_addresses(sms)
    for email in emails_iter:
        if email.group() in sms_email_blacklist_path.values:
            return JSONResponse(
                content=Response(type="spam", spam_percent=1.0).to_dict()
            )

    # Check for blacklisted phone numbers in the SMS
    phones_iter = detect_general_phone_number(sms)
    for phone in phones_iter:
        if phone.group() in sms_phone_blacklist_path.values:
            return JSONResponse(
                content=Response(type="spam", spam_percent=1.0).to_dict()
            )

    # Check for blacklisted URLs in the SMS
    detected_urls = detect_urls(sms)
    detected_urls_predictions = []
    for url in detected_urls:
        url_domain_path = extract_domain_path(url.group())
        if any(
            url_domain_path.startswith(blacklisted_domain_path[0])
            for blacklisted_domain_path in url_blacklist_path.values
        ):
            return JSONResponse(
                content=Response(type="spam", spam_percent=1.0).to_dict()
            )

        # Extract features from the URL and predict if it's spam
        url_features = np.array(
            URLFeatureExtractor(url.group()).getFeaturesList()
        ).reshape(1, 30)
        url_prediction = app_models.url.predict(url_features)[0]
        url_spam_percent = app_models.url.predict_percent(url_features)[0]
        detected_urls_predictions.append((url_prediction, url_spam_percent))

    # Predict if the SMS is spam
    prediction = app_models.sms.predict(np.array([sms]))[0]
    spam_percent = app_models.sms.predict_percent(np.array([sms]))[0]

    # Combine URL and SMS predictions
    if len(detected_urls_predictions) > 0:
        url_spam_predictions = [pred == -1 for pred, _ in detected_urls_predictions]
        url_spam_percents = [percent for _, percent in detected_urls_predictions]

        combined_prediction = 1 if any(url_spam_predictions) else prediction
        combined_spam_percent = max(url_spam_percents + [spam_percent])
    else:
        combined_prediction = prediction
        combined_spam_percent = spam_percent

    return JSONResponse(
        content=Response(
            type="spam" if combined_prediction == 1 else "ham",
            spam_percent=float(combined_spam_percent),
        ).to_dict()
    )


# Endpoint to classify URLs
@app.post("/url/classify")
async def classify_url(request: URLRequest):
    url = request.url
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    # Check if the URL is blacklisted
    url_domain_path = extract_domain_path(url)
    if any(
        url_domain_path.startswith(blacklisted_domain_path[0])
        for blacklisted_domain_path in url_blacklist_path.values
    ):
        return JSONResponse(content=Response(type="spam", spam_percent=1.0).to_dict())

    # Extract features from the URL and predict if it's spam
    features = np.array(URLFeatureExtractor(url).getFeaturesList()).reshape(1, 30)
    prediction = app_models.url.predict(features)[0]
    spam_percent = app_models.url.predict_percent(features)[0]

    return JSONResponse(
        content=Response(
            type="spam" if prediction == -1 else "ham",
            spam_percent=float(spam_percent),
        ).to_dict()
    )


# Endpoint to serve static files
@app.get("/{path:path}")
async def serve_static(path: str = ""):
    dist_path = "./spam-detector-web/dist/spam-detector-web/browser"
    full_path = os.path.join(dist_path, path)
    if path and os.path.exists(full_path):
        return FileResponse(full_path)
    return FileResponse(os.path.join(dist_path, "index.html"))


# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, reload=True)
