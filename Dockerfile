FROM continuumio/miniconda3:4.7.12

ARG LOG_DIR
ARG MODEL_TYPE

RUN apt-get update -qq

COPY . /

RUN make env
RUN make train

EXPOSE 8080

CMD ["make app"]
