FROM continuumio/miniconda3:4.7.12

ARG LOG_DIR

RUN apt-get update -qq

COPY . /
RUN ./startup_script.sh

EXPOSE 8080

CMD ["./run.sh"]
