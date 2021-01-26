FROM python:3.8-buster

WORKDIR /opt

ADD / /opt

RUN pip install -r requirement.txt

ENTRYPOINT [ "python", "u","/opt/flask_ex1.py","500"]