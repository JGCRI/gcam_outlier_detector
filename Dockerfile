FROM ubuntu:22.04 AS data_sci
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get install -y \ 
    openjdk-11-jdk-headless python3 python3-pip libboost-python-dev libboost-numpy-dev \
    openjdk-11-jre-headless libtbb12 rsync git
RUN apt-get -y update && apt -y upgrade
RUN pip install pandas numpy matplotlib
RUN git clone https://github.com/JGCRI/gcamreader.git /gcamreader
RUN pip install /gcamreader
COPY jovyan/anamoly_detector /home/jovyan/anamoly_detector