.PHONY: env data

#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL=/bin/bash
PROJECT_NAME = attention-lstm-sentiment-analysis
PROJECT_DIR = $(shell pwd)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up conda environment and install dependencies
env:
	if type "conda" > /dev/null;
	then
		conda env create -f environment.yaml --force
		conda activate $(PROJECT_NAME)
	else
		echo "Please install conda in your environment!"
	fi

## remove conda environment
remove_env:
ifeq (True,$(HAS_PYENV))
		@echo ">>> Detected pyenv, removing pyenv version."
		pyenv local ${CONDA_VERSION} && rm -rf ~/.pyenv/versions/${CONDA_VERSION}/envs/$(PROJECT_NAME)
else
		@echo ">>> Removing conda environemnt"
		conda remove -n $(PROJECT_NAME) --all
endif

install_src:
	pip install -e .

## build and push docker image for clockwork to google container registry
image:
	bash ./scripts/build_image.sh $(PROJECT_NAME)

## Install docker if missing
get_docker:
ifeq (True,$(HAS_DOCKER))
		@echo ">>> Docker already installed"
else
		@echo ">>> Installing docker"
		sh scripts/get_docker.sh
endif

## start tf serving in local docker container
serving_up: serving_down
	bash ./scripts/serving_up.sh $(CONFIG_DIR) ${BINARIES}/mlflow ${RUN_ID}

## stop and remove docker container if running
serving_down: get_docker
	docker stop sauron_tf_serving || true && docker rm sauron_tf_serving || true

## start tensorboard
tensorboard:
	bash ./scripts/tensorboard.sh $(CONFIG_DIR) ${BINARIES} ${RUN_ID}

# open what-if tool (tf serving and tensorboard should by already running)
what_if:
	bash ./scripts/what_if_tool.sh $(CONFIG_DIR) ${BINARIES}/mlflow ${RUN_ID}

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Lint using flake8
lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

## Generate clockwork pipeline config
pipeline:
	bash ./scripts/make_pipeline.sh PROJECT_NAME=${PROJECT_NAME} CONFIG_DIR=${CONFIG_DIR} DOCKER_TAG=${DOCKER_TAG} PUSH=${push}

## Push to clockwork scheduler repo
push_schedule:
	bash ./scripts/push_schedule.sh CONFIG_DIR=${CONFIG_DIR}

update_all_pipelines:
	for config_dir in $(shell ls -d env/models/*/*/*/*.*.* | grep -v test); do \
		bash ./scripts/make_pipeline.sh PROJECT_NAME=${PROJECT_NAME} CONFIG_DIR=$${config_dir} DOCKER_TAG=${DOCKER_TAG} PUSH=${push} ; \
	done

## deprecate this experiment
deprecate:
	${BINARIES}/python -m src.models.make_deprecate $(CONFIG_DIR)

## Run doc tests and unit tests
test:
	bash ./test/test_runner.sh NOSETESTS_EXECUTABLE=${BINARIES}/nosetests SAVED_MODELS_CLI_PATH=${BINARIES}/saved_model_cli

## Run steamlit model evaluation app
streamlit_app:
	streamlit run src/visualisation/streamlit_app.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



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
