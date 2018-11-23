# Dockerfile to create env: build using `datmo env build`
# [1] Base environment to start from:
# Find more at https://hub.docker.com/u/datmo/
FROM datmo/python-base:cpu-py35

RUN apt-get update; apt-get install -y python3-pip python3-numpy python3-scipy python3-wheel
ADD requirements.txt /
RUN pip3 install -r /requirements.txt


RUN mkdir /code;mkdir /code/models
COPY ./python_api.py ./letter.py /code/
COPY ./models/* /code/models/


WORKDIR /code

CMD python3 /code/python_api.py
