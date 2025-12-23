.PHONY: up down build logs shell test lint clean reset load

# Infrastructure
up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build --progress=plain

restart: down up

logs:
	docker compose logs -f app

shell:
	docker compose exec app /bin/bash

# Data & Testing
load:
	@echo "Loading dataset..."
	docker compose exec app python -m src.scripts.load_data

reset:
	@echo "Resetting database..."
	docker compose down -v
	docker compose up -d
	@echo "Waiting for DB..."
	sleep 5
	make load
