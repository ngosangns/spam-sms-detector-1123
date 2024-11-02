import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models import (
    # BERTClassifier,
    SVMClassifier,
    # NaiveBayesClassifier,
    # RandomForestClassifier,
    # LogisticRegressionClassifier,
    # KNNClassifier,
    # GradientBoostingClassifier,
)

RESULT_DIR = "./ml-models"
STATIC_PATH = "./web/dist"

# Load the BERT model
model = SVMClassifier(RESULT_DIR)
model.load_model()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SMSRequest(BaseModel):
    sms: str


@app.post("/classify")
async def classify_sms(request: SMSRequest):
    sms = request.sms
    if not sms:
        raise HTTPException(status_code=400, detail="No SMS provided")

    # Predict using the model
    prediction = model.predict(sms)
    sms_type = "spam" if prediction[0] == 1 else "ham"

    return JSONResponse(content={"type": sms_type})


@app.get("/{path:path}")
async def serve_static(path: str = ""):
    full_path = os.path.join(STATIC_PATH, path)
    if path and os.path.exists(full_path):
        return FileResponse(full_path)
    return FileResponse(os.path.join(STATIC_PATH, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, reload=True)
