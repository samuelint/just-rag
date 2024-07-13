.PHONY: install
test:
	poetry install

.PHONY: test
test:
	poetry run pytest
