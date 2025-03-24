up:
	docker compose up -d

stop:
	docker compose stop

down:
	docker compose down


format:
	black src
	isort src


freeze:
	uv pip freeze > requirements.txt


step1:
	uv run src/news_chatbot/pipelines/step1.py

step2:
	uv run src/news_chatbot/pipelines/step2_1.py