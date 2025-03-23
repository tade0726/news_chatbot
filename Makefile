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


query_news:
	uv run src/news_chatbot/pipelines/search_news.py