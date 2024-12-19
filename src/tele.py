import logging

import numpy as np
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.app_models import init_app_models
from src.models.url_feature_extractor import URLFeatureExtractor

# Bot username: @sms_spam_detector_1123_bot
BOT_TOKEN = "7546000010:AAFM8d-aHENDBuWPkDGgudy2U9ia0DONap8"

app_models = init_app_models()


# Set up logging for debugging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def handle_sms(text: str) -> bool:
    return "sms" in text.lower()


async def receive_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    response = handle_sms(text)
    await update.message.reply_text(str(response).lower())


async def handle_detect_sms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    sms_content = text[5:]
    prediction = app_models.sms.predict(np.array([sms_content]))[0]
    spam_percent = app_models.sms.predict_percent(np.array([sms_content]))[0]
    response = f"Phân loại: {'Spam' if prediction == 1 else 'Thông thường'}. Tỉ lệ spam: {(spam_percent*100):.2f}%"
    await update.message.reply_text(str(response))


async def handle_detect_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    url_content = text[5:]
    features = np.array(URLFeatureExtractor(url_content).getFeaturesList()).reshape(
        1, 30
    )
    prediction = app_models.url.predict(features)[0]
    spam_percent = app_models.url.predict_percent(features)[0]
    response = f"Phân loại: {'Spam' if prediction == -1 else 'Thông thường'}. Tỉ lệ spam: {(spam_percent*100):.2f}%"
    await update.message.reply_text(str(response))


def main():
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, receive_message)
    )
    application.add_handler(CommandHandler("sms", handle_detect_sms))
    application.add_handler(CommandHandler("url", handle_detect_url))
    application.run_polling()


if __name__ == "__main__":
    main()
