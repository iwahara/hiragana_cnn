FROM python:3.6-jessie

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt

WORKDIR /var
RUN mkdir models
COPY ./prepare_dataset.py ./
COPY ./hiragana_cnn.py ./
COPY ./docker-entrypoint.sh ./
