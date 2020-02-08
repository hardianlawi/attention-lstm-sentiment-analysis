FROM continuumio/miniconda3:4.7.12

ARG PROJECT_NAME
ARG LOG_DIR
ARG MODEL_TYPE

WORKDIR ${PROJECT_NAME}
COPY . /opt/${PROJECT_NAME}

RUN apt-get update -qq && apt-get install -y make cmake

RUN make env
RUN make train

EXPOSE 8080

CMD ["./run.sh"]
