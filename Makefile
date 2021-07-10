.PHONY: lint pretty run setup test


setup:
	poetry install

test:
	poetry run pytest

run:
	poetry run solver 1

pretty:
	poetry run black .

lint:
	poetry run flake8 .

pr: pretty lint test

vis:
	open ./visualizer/index.html
