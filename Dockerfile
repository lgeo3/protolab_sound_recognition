FROM continuumio/anaconda
USER root

# Install some dev tools
RUN apt-get update
RUN apt-get install -y wget gcc g++ make alsa-base alsa-utils

RUN wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.25.tar.gz && tar xvzf libsndfile-1.0.25.tar.gz && cd libsndfile-1.0.25 && ./configure --prefix=/usr && make && make install

ADD . protolab_sound_recognition 
RUN cd protolab_sound_recognition && pip install -r requirements.txt  && python setup.py develop
