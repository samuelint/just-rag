.PHONY: install
test:
	poetry install

# PROJECT="another-project" make rename
.PHONY: rename
rename:
	@if [ -z "$(PROJECT)" ]; then \
		echo "Error: PROJECT variable not set"; \
		exit 1; \
	fi;

	# Clean and formatted project variables
	$(eval CLEAN_PROJECT_NAME := $(shell echo "$(PROJECT)" | tr ' -' '_'))
	$(eval PROJECT_TOML_NAME := $(shell echo "$(PROJECT)" | tr ' ' '-'))

	# Rename directory
	@echo "Renaming directory langgraph_agent_template to $(CLEAN_PROJECT_NAME)..."
	@mv langgraph_agent_template $(CLEAN_PROJECT_NAME)

	# Update pyproject.toml
	@echo "Updating project name in pyproject.toml..."
	@sed -i '' 's/langgraph-agent-template/$(PROJECT_TOML_NAME)/g' pyproject.toml

	# Update all .py files
	@echo "Updating project name in .py files..."
	@find . -type f -name "*.py" -print0 | xargs -0 sed -i '' 's/langgraph_agent_template/$(CLEAN_PROJECT_NAME)/g'

	@echo "Rename complete!"

.PHONY: test
test:
	poetry run pytest
