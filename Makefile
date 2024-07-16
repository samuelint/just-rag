export PYTHONPATH=$(shell pwd)

.PHONY: install
install:
	poetry install

.PHONY: test
test:
	poetry run pytest
