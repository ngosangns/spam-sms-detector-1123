import os
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from models.sms_rnn_classifier import SingletonSMSRNNClassifier
from models.url_feature_extractor import URLFeatureExtractor
from models.url_ml_catboost_classifier import SingletonURLMLCatBoostClassifier

model_dir = "./trained_models"

sms_model = SingletonSMSRNNClassifier(model_dir)
sms_model.load()

url_model = SingletonURLMLCatBoostClassifier(model_dir)
url_model.load()

app = FastAPI()


class SMSRequest(BaseModel):
    sms: str


@app.post("/sms/classify")
async def classify_sms(request: SMSRequest):
    sms = request.sms
    if not sms:
        raise HTTPException(status_code=400, detail="No SMS provided")

    prediction = sms_model.predict(np.array([sms]))[0]
    spam_percent = sms_model.predict_percent(np.array([sms]))[0]

    return JSONResponse(
        content={
            "type": "spam" if prediction == 1 else "ham",
            "spam_percent": float(spam_percent),
        }
    )


class URLRequest(BaseModel):
    url: str


@app.post("/url/classify")
async def classify_url(request: URLRequest):
    url = request.url
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    features = np.array(URLFeatureExtractor(url).getFeaturesList()).reshape(1, 30)
    prediction = url_model.predict(features)[0]
    spam_percent = url_model.predict_percent(features)[0]

    return JSONResponse(
        content={
            "type": "spam" if prediction == -1 else "ham",
            "spam_percent": float(spam_percent),
        }
    )


@app.get("/{path:path}")
async def serve_static(path: str = ""):
    dist_path = "./spam-detector-web/dist/spam-detector-web/browser"
    full_path = os.path.join(dist_path, path)
    if path and os.path.exists(full_path):
        return FileResponse(full_path)
    return FileResponse(os.path.join(dist_path, "index.html"))


if __name__ == "__main__":
    uvicorn.run(app, reload=True)
