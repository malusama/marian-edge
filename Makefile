.DEFAULT_GOAL := help

.PHONY: help fmt lint test check build-macos model package docker-build docker-up

help: ## Show common developer commands
	@awk 'BEGIN {FS = ":.*## "; print "Marian MLX developer commands:"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

fmt: ## Format Rust source
	cargo fmt --all

lint: ## Run Clippy with warnings denied
	cargo clippy --workspace --all-targets --locked -- -D warnings

test: ## Run the portable unit and API tests
	cargo test --workspace --locked

check: ## Run the portable Rust checks
	cargo fmt --all --check
	cargo clippy --workspace --all-targets --locked -- -D warnings
	cargo test --workspace --locked

build-macos: ## Build the native Apple Silicon MLX runtime
	scripts/build-mlx.sh
	scripts/build-release.sh

model: ## Download, verify, and convert the en-to-zh model
	scripts/prepare-enzh-model.sh

package: ## Create the native macOS release archive
	scripts/package-macos.sh

docker-build: ## Build the CPU-only Docker image for the host architecture
	docker build --tag marian-mlx:cpu .

docker-up: ## Start the CPU-only service with Docker Compose
	docker compose up --detach
