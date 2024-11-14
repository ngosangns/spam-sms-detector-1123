import os
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app_dto import Response, SMSRequest, URLRequest
from app_models import init_app_models
from models.url_feature_extractor import URLFeatureExtractor

app_models = init_app_models()

app = FastAPI()


@app.post("/sms/classify")
async def classify_sms(request: SMSRequest):
    sms = request.sms
    if not sms:
        raise HTTPException(status_code=400, detail="No SMS provided")

    prediction = app_models.sms.predict(np.array([sms]))[0]
    spam_percent = app_models.sms.predict_percent(np.array([sms]))[0]

    return JSONResponse(
        content=Response(
            type="spam" if prediction == 1 else "ham",
            spam_percent=float(spam_percent),
        ).to_dict()
    )


@app.post("/url/classify")
async def classify_url(request: URLRequest):
    url = request.url
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    features = np.array(URLFeatureExtractor(url).getFeaturesList()).reshape(1, 30)
    prediction = app_models.url.predict(features)[0]
    spam_percent = app_models.url.predict_percent(features)[0]

    return JSONResponse(
        content=Response(
            type="spam" if prediction == -1 else "ham",
            spam_percent=float(spam_percent),
        ).to_dict()
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
