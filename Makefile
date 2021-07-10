# https://stackoverflow.com/a/14061796
# If the first argument is "run"...
ifeq (run,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif

.PHONY: lint pretty run setup test types


setup:
	poetry install

test:
	poetry run pytest

run:
	poetry run solver $(RUN_ARGS)

pretty:
	poetry run isort .
	poetry run black .

lint:
	poetry run flake8 .

types:
	poetry run mypy .

pr: pretty lint test

vis:
	open ./visualizer/index.html
