FROM continuumio/miniconda3:4.8.2

ARG LOG_DIR
ARG MODEL_TYPE

RUN apt-get update -qq && apt-get install -y make cmake

COPY . /

RUN make env
RUN make train

EXPOSE 8080

CMD ["make app; make test"]
