i:
	pip install -r requirements.txt

dev:
	conc "cd spam-detector-web; yarn watch" "fastapi dev app.py"

serve:
	conc "cd spam-detector-web; yarn build" "fastapi dev app.py"

mc:
	clear
	python sms_compare.py

uc:
	clear
	python url_compare.py