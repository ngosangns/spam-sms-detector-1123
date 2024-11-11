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

sms_model = SingletonSMSMLSVMClassifier("./trained_models")
sms_model.load()

url_model = SingletonURLMLGradientBoostingClassifier("./trained_models")
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

    # y_pro_phishing = self.model.predict_proba(obj)[0, 0]
    # y_pro_non_phishing = self.model.predict_proba(obj)[0, 1]

    prediction = url_model.predict(extracted_features)[0]
    url_type = "spam" if prediction == -1 else "ham"

    return JSONResponse(content={"type": url_type})


@app.get("/{path:path}")
async def serve_static(path: str = ""):
    full_path = os.path.join("./web/dist", path)
    if path and os.path.exists(full_path):
        return FileResponse(full_path)
    return FileResponse(os.path.join("./web/dist", "index.html"))


if __name__ == "__main__":

    uvicorn.run(app, reload=True)
