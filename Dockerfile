# FROM ubuntu:noble-20240114
FROM ubuntu:jammy-20240111


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.8 python3-distutils python3-pip python3-apt

RUN apt-get install -y openjdk-8-jdk
RUN apt-get install -y vim

RUN java -version

# SET ENV VARIABLES

ENV JAVA_HOME /usr
ENV LANG en_US.utf8
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3
RUN export JAVA_HOME
RUN java -version


# COPY CODE
RUN mkdir /code
WORKDIR /code
RUN echo "$PWD"
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install cython
RUN pip3 install --no-cache-dir -r requirements.txt


COPY . .

RUN ["chmod", "-R", "u=rwX,g=rwX", "/tmp"]
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /code
USER appuser

# ADD --chown=appuser:appuser code /code
USER appuser

ENTRYPOINT ["python3","src/Wine-quality-Inference.py"]
CMD []
