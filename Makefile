i:
	pip install -r requirements.txt

dev:
	cd spam-detector-web; yarn watch &
	fastapi dev app.py &
	wait

mc:
	clear
	python sms_compare.py

uc:
	clear
	python url_compare.py