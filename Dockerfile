FROM python:3.10.5-slim-buster

LABEL maintainer.email="maksim.carkov.201300@gmail.com"
LABEL maintainer.git="s1Sharp"

EXPOSE 5000

RUN apt update && apt upgrade -y

WORKDIR /image-generator

COPY . .

RUN pip install -r requirements.txt
