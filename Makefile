run:
	uvicorn src.main:app --reload

docker-up:
	docker-compose up -d --build

test:
	pytest
