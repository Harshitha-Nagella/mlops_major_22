#FROM ubuntu:latest
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./api /exp/
COPY ./utils /exp/utils
ADD ./models/ /exp/models/
COPY ./requirements.txt /exp/requirements.txt
ADD ./results /exp/results
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD ["python3", "./plots_graph.py","SVM","42"]