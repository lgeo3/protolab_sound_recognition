Sound event recognition using python
--------------------------------------

The main objective of this project is to provide sound recognition for sound like door bell, phone ring, fire alarm, etc.
It uses MFCC features and scikit-learn for sound classification.

Build travis: 

.. image:: https://travis-ci.org/laurent-george/protolab_sound_recognition.svg?branch=master
    :target: https://travis-ci.org/laurent-george/protolab_sound_recognition


Sound dataset
==============

A simple sound dataset including 6 kinds of sounds (applause, deskbell, doorbell, firealarm, tactac mouse sound, whistle) can be downloaded here: `aldebaran 6 sounds dataset <https://www.dropbox.com/s/ekldjq8o1wfhcq1/dataset_aldebaran_6sounds.tar.gz?dl=0>`__.


Installation
==============

On ubuntu:
-----------

.. code:: bash

	sudo apt-get install libsndfile1-dev python-numpy python-scipy python-sklearn python-pandas
	pip install -r requirements.txt  --user
	python setup.py develop --user


Running flask service:
------------------------

.. code:: bash

	mkdir data
	python flask_restful_app/api.py -p "/var/innov/data/sounds/dataset/*.wav"

where "/var/innov/data/sounds/dataset/\*.wav" contains your dataset wav file (name of class in the filename).

All files sended to the server are saved under directory data/




License
=========

Please see LICENCE.txt in this directory.
