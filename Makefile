COMPOSE ?= docker compose
PORT    ?= 8000
export PORT

.DEFAULT_GOAL := help
.PHONY: help env up down build rebuild restart logs ps shell test clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example if it does not exist
	@test -f .env || (cp .env.example .env && echo "Created .env - set GEMINI_API_KEY in it")

up: env ## Build the image and start the app in the background
	$(COMPOSE) up --build -d app
	@echo "pdfchunk is running at http://localhost:$(PORT)"

down: ## Stop and remove the containers (keeps processed data)
	$(COMPOSE) --profile test down --remove-orphans

build: ## Build the images without starting them
	$(COMPOSE) --profile test build

rebuild: env ## Rebuild from scratch (no cache) and start
	$(COMPOSE) build --no-cache app
	$(COMPOSE) up -d --force-recreate app

restart: ## Restart the app container (picks up .env changes)
	$(COMPOSE) up -d --force-recreate app

logs: ## Follow the app logs
	$(COMPOSE) logs -f app

ps: ## Show container status
	$(COMPOSE) ps

shell: ## Open a shell inside the running app container
	$(COMPOSE) exec app bash

test: ## Run the test suite inside Docker
	$(COMPOSE) --profile test run --rm --build test

clean: ## Stop everything and delete containers, images AND the data volume
	$(COMPOSE) --profile test down --volumes --remove-orphans --rmi local
