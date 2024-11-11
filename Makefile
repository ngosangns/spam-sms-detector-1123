i:
	pip install -r requirements.txt

dev:
	fastapi dev app.py

mc:
	clear
	python sms_compare.py

uc:
	clear
	python url_compare.py