up:
	docker-compose up -d --build
down:
	docker-compose down
load:
	docker-compose exec web python -m src.scripts.load_data
bench:
	docker-compose exec web python -m src.scripts.benchmark
test:
	docker-compose exec web pytest
