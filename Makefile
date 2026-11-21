.PHONY: up down logs ps test restart clean

up: ## Build and start backend (:8000) and frontend (:3000)
	@test -f .env || (cp .env.example .env && echo ">> Created .env from .env.example. Put your real LLM_API_KEYS in it (or set MOCK_LLM=true), then run make up again." && exit 1)
	docker compose up -d --build
	@echo ">> Frontend: http://localhost:3000   Backend: http://localhost:8000/health"

down: ## Stop the stack (the SQLite volume is kept)
	docker compose down

restart: down up

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

test: ## Run backend tests locally (needs Python 3.11+)
	cd backend && pip install -q -r requirements-dev.txt && python -m pytest -q

clean: ## Stop and delete the SQLite volume too
	docker compose down -v
