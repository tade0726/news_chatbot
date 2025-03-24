up:
	docker compose up -d

stop:
	docker compose stop

down:
	docker compose down


format:
	black src

freeze:
	uv pip freeze > requirements.txt


step1:
	uv run src/news_chatbot/pipelines/step1.py

step2_1:
	uv run src/news_chatbot/pipelines/step2_1.py

step2_2:
	uv run src/news_chatbot/pipelines/step2_2.py

step3:
	uv run src/news_chatbot/pipelines/step3.py

step4:
	uv run src/news_chatbot/pipelines/step4.py
