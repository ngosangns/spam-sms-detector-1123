import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

TOKEN = "7546000010:AAFM8d-aHENDBuWPkDGgudy2U9ia0DONap8"

# Set up logging for debugging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_sms(text: str) -> bool:
    return "sms" in text.lower()

async def receive_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    response = handle_sms(text)
    await update.message.reply_text(str(response).lower())

def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, receive_message))
    application.run_polling()

if __name__ == '__main__':
    main()
