import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sms.models import (
    SMSSVMClassifier,
    # BERTClassifier,
    # NaiveBayesClassifier,
    # RandomForestClassifier,
    # LogisticRegressionClassifier,
    # KNNClassifier,
    # GradientBoostingClassifier,
)
from sms.utils import predict
from url.models import (
    # BERTClassifier,
    # SVMClassifier,
    # NaiveBayesClassifier,
    # RandomForestClassifier,
    # LogisticRegressionClassifier,
    # KNNClassifier,
    GradientBoostingClassifier as URLGradientBoostingClassifier,
)

load_dotenv()

RESULT_DIR = os.getenv("RESULT_DIR")
STATIC_PATH = os.getenv("STATIC_PATH")

sms_model = SMSSVMClassifier(RESULT_DIR)
sms_model.load()

url_model = URLGradientBoostingClassifier(RESULT_DIR)
url_model.load_model()

app = FastAPI()


class SMSRequest(BaseModel):
    sms: str


@app.post("/sms/classify")
async def classify_sms(request: SMSRequest):
    sms = request.sms
    if not sms:
        raise HTTPException(status_code=400, detail="No SMS provided")

    prediction = predict(sms_model.model, sms)
    sms_type = "spam" if prediction[0] == 1 else "ham"

    return JSONResponse(content={"type": sms_type})


class URLRequest(BaseModel):
    url: str


@app.post("/url/classify")
async def classify_url(request: URLRequest):
    url = request.url
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    prediction = url_model.predict(url)
    url_type = "spam" if prediction == -1 else "ham"

    return JSONResponse(content={"type": url_type})


@app.get("/{path:path}")
async def serve_static(path: str = ""):
    full_path = os.path.join(STATIC_PATH, path)
    if path and os.path.exists(full_path):
        return FileResponse(full_path)
    return FileResponse(os.path.join(STATIC_PATH, "index.html"))


if __name__ == "__main__":

    uvicorn.run(app, reload=True)
