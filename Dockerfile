FROM continuumio/miniconda3:4.7.12

ARG PROJECT_NAME
ARG LOG_DIR
ARG MODEL_TYPE

COPY . /${PROJECT_NAME}
WORKDIR /${PROJECT_NAME}

ENV LOG_DIR=/${PROJECT_NAME}/${LOG_DIR}
ENV MODEL_TYPE=${MODEL_TYPE}

RUN apt-get update -qq && apt-get install -y make cmake vim curl

RUN make env
RUN make train

EXPOSE 8080

RUN ["chmod", "+x", "run.sh"]
CMD ["./run.sh"]
