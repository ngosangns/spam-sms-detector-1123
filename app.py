import os
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from models.sms_ml_svm_classifier import SingletonSMSMLSVMClassifier
from models.url_feature_extractor import URLFeatureExtractor
from models.url_ml_gradient_boosting_classifier import (
    SingletonURLMLGradientBoostingClassifier,
)

model_dir = "./trained_models"

sms_model = SingletonSMSMLSVMClassifier(model_dir)
sms_model.load()

url_model = SingletonURLMLGradientBoostingClassifier(model_dir)
url_model.load()

app = FastAPI()


class SMSRequest(BaseModel):
    sms: str


@app.post("/sms/classify")
async def classify_sms(request: SMSRequest):
    sms = request.sms
    if not sms:
        raise HTTPException(status_code=400, detail="No SMS provided")

    prediction = sms_model.predict(np.array([sms]))
    sms_type = "spam" if prediction[0] == 1 else "ham"

    return JSONResponse(content={"type": sms_type})


class URLRequest(BaseModel):
    url: str


@app.post("/url/classify")
async def classify_url(request: URLRequest):
    url = request.url
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    obj = URLFeatureExtractor(url)
    extracted_features = np.array(obj.getFeaturesList()).reshape(1, 30).tolist()

    url_type = (
        "spam" if url_model.predict(np.array(extracted_features))[0] == -1 else "ham"
    )
    spam_percent = url_model.predictPercent(np.array(extracted_features))

    return JSONResponse(content={"type": url_type, "spam_percent": spam_percent})


@app.get("/{path:path}")
async def serve_static(path: str = ""):
    dist_path = "./spam-detector-web/dist/spam-detector-web/browser"
    full_path = os.path.join(dist_path, path)
    if path and os.path.exists(full_path):
        return FileResponse(full_path)
    return FileResponse(os.path.join(dist_path, "index.html"))


if __name__ == "__main__":

    uvicorn.run(app, reload=True)
