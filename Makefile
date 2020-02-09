.PHONY: env data test

#################################################################################
# GLOBALS                                                                       #
#################################################################################

include versions.txt

SHELL=/bin/bash
PROJECT_NAME = attention-lstm-sentiment-analysis
PROJECT_DIR = $(shell pwd)
CONDA_ROOT=$(shell conda info --root)

ifeq (,$(shell which pyenv))
	HAS_PYENV=False
	CONDA_ROOT=$(shell conda info --root)
ifeq (True,$(shell if [ ! -d ${CONDA_ROOT}/envs/${PROJECT_NAME} ]; then echo True; fi))
	# use conda root if env is missing (when running on Docker)
	BINARIES = ${CONDA_ROOT}/bin
else
	BINARIES = ${CONDA_ROOT}/envs/${PROJECT_NAME}/bin
endif
else
	HAS_PYENV=True
	CONDA_VERSION=$(shell echo $(shell pyenv version | awk '{print $$1;}') | awk -F "/" '{print $$1}')
	BINARIES = $(HOME)/.pyenv/versions/${CONDA_VERSION}/envs/${PROJECT_NAME}/bin
endif

ifeq (,$(shell which docker))
	HAS_DOCKER=False
else
	HAS_DOCKER=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

check_path:
	echo "conda path: $(CONDA_ROOT)"
	echo "python path: $(BINARIES)"

## Set up conda environment and install dependencies
env:
ifeq (, $(shell which conda))
	echo "Please install conda in your environment!"
else
	conda env create -n $(PROJECT_NAME) -f environment.yaml --force
endif

remove_env:
	conda remove -n $(PROJECT_NAME) --all

install_kernel:
	$(BINARIES)/python -m ipykernel install --user --name $(PROJECT_NAME)

## Install docker if missing
install_docker:
ifeq (True,$(HAS_DOCKER))
	@echo ">>> Docker already installed"
else
	@echo ">>> Installing docker"
	$(SHELL) scripts/get_docker.sh
endif

train:
	$(BINARIES)/python -m src.train $(LOG_DIR) $(MODEL_TYPE)

app:
	$(BINARIES)/python -m src.app $(LOG_DIR) $(MODEL_TYPE)

build_app:
ifeq (True,$(HAS_DOCKER))
	docker build -t sentiment-app-$(MODEL_TYPE):$(APP_VERSION) . --build-arg PROJECT_NAME=$(PROJECT_NAME) --build-arg LOG_DIR=model --build-arg MODEL_TYPE=$(MODEL_TYPE)
else
	@echo ">>> Please install docker"
endif

run_app:
	docker run -p 8080:8080 --detach sentiment-app-$(MODEL_TYPE):$(APP_VERSION)

test_app:
	$(eval CONTAINER_ID = $(shell docker run -p 8080:8080 --detach sentiment-app-$(MODEL_TYPE):$(APP_VERSION) | cut -c 1-12))
	mkdir -p /tmp/model
	docker cp $(CONTAINER_ID):/$(PROJECT_NAME)/model/test_requests.json /tmp/model/test_requests.json
	$(BINARIES)/python -m src.test_api /tmp/model
	rm -r /tmp/model
	docker kill $(CONTAINER_ID)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
