Sound event recognition using python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: https://travis-ci.org/laurent-george/protolab_sound_recognition.svg?branch=master
    :target: https://travis-ci.org/laurent-george/protolab_sound_recognition

The main objective of this project is to provide sound recognition for sound like door bell, phone ring, fire alarm, etc.
It uses MFCC features and scikit-learn for sound classification.


Installation:
==============

A first requirements is libsndfile. To install it:

.. code:: bash

    wget http://www.mega-nerd.com/libsndfile/files/libsndfile-1.0.25.tar.gz
    tar xvzf libsndfile-1.0.25.tar.gz
    cd libsndfile-1.0.25
    ./configure
    make
    sudo make install

Then you can clone the directory:

.. code:: bash

    git clone https://github.com/laurent-george/protolab_sound_recognition.git
    cd protolab_sound_recognition
    pip install -r requirements.txt  --user
    python setup.py develop --user


Checking installation:
----------------------------------
In order to check that the installation is correct you can run the following test:

.. code::

    py.test tests/test_bell_detection.py

Usage:
=======

Sound dataset:
-----------------
In order to recognize sounds you need to have a set of labeled sounds. We provide two example datasets that you can use freely.

`Simple sound dataset <https://www.dropbox.com/s/ekldjq8o1wfhcq1/dataset_aldebaran_6sounds.tar.gz?dl=0>`__ : 6 sound types (applause, deskbell, doorbell, firealarm, tactac mouse sound, whistle). `A more complete sound dataset <https://www.dropbox.com/s/8t427pyszfhkfm4/dataset_aldebaran_allsounds.tar.gz?dl=0>`__ : 21 sound types  (ApplauseLight, BlowNose, ClapHand, DeskBell, DoorBell01, DoorBell02, FakeSneeze, FireAlarmFr, HumanCaressHead, HumanScratchHead, Laugh, NoisePaper, RobotNoiseMoving, RobotNoisePushed, ShutDoor, SmokeDetector, TacTac, ToyGiraffe, ToyMaracas, ToyPig, Whistle).

Files in the sound dataset should be wav file with name prefix describing the class of the sound as in `Applause-014-NaoAlex.wav` which is `Applause class`.

Running flask service:
------------------------

To facilitate the usage of the sound_recognition module there is a flask application that provide a rest api. You can use it to run a webapp on a remote server to provide sound recognition capability to clients.

.. code:: bash

	mkdir data
	python flask_restful_app/api.py -w "path_to_the_sound_dataset/*.wav"
        eg for the complete sound dataset with 21 subfolders:
        python flask_restful_app/api.py -w "path_to_the_sound_dataset/*/*.wav"


where "path_to_the_sound_dataset\/*.wav" contains your dataset wav file.

Then you have a running web app that you can access from python for example:

.. code:: python

    import request
    wav_file = "example.wav"
    server_ip = "127.0.0.1"
    webservice_pageaddress = "http://%s:5000/api/ClassifySoundFile" % server_ip

    audio_file = open(wav_file, 'r')
    device_type = ''

    payload = {'content_classification': 'unknown', 'device_type':device_type}
    files = {'audio_file':audio_file}

    r = requests.post(webservice_pageaddress, data=payload, files=files)


The return value is a dictionary with keys *classif* and *filename*. Classif field contains a list of classification results, each items in the list is composed of [*timestamp_start*, *timestamp_stop*, *sound_type_detected*, *classifier_confidence*, *score confidence*]. The filename field corresponds to the name of file as it saved on the server (it is usefull to report a bug on a specific file for instance, so we can investigate on a specific sound).

The score is a scaled value that allow you to know if you can be confident that a sound occurs: if score > 1 you can be confident.


Here is an example of result:

.. code:: python

    import pprint
    pprint.pprint(r)
    {'classif': [[0.0, 1.0, 'RobotNoiseMoving', 0.2403505903474301, 0.49162620752883424],
                 [0.5, 1.5, 'DeskBell', 0.6298477177974425, 5.001747617851262],
                 [1.0, 2.0, 'DeskBell', 0.8441221425138888, 6.703343992828691],
                 [1.5, 2.5, 'DeskBell', 0.902478142033486, 7.166760741571032],
                 [2.0, 3.0, 'DeskBell', 0.6344029251674168, 5.037921437280603],
                 [2.5, 3.5, 'RobotNoiseMoving', 0.18166274340036412, 0.3715828842280175],
                 [3.0, 4.0, 'RobotNoiseMoving', 0.4039192099803734, 0.8261983840507637],
                 [3.5, 4.5, 'ShutDoor', 0.1879766200859678, 0.30759810559522]],
     'filename': '2015_09_11-10h57m26_ROBOT_unknown_2991.wav'}

Here based on the score, we can be confident that there is a *DeskBell* sound between 0.5s and 3.0s. And if we need a more precise window, the highest detection is between 1.5s and 2.5s (i.e highest score).


Using the module directly:
---------------------------

Of course you can also use the module directly without flask.

.. code:: python

    dataset_path = 'PATH_TO_THE_DATASET'
    file_regexp = os.path.join(dataset_path, '*.wav')
    files = glob.glob(file_regexp)
    sound_classification_obj = classification_service.SoundClassification(wav_file_list=files, calibrate_score=True)
    sound_classification_obj.learn()
   wav_file_path = '/tmp/test.wav'
    res = sound_classification_obj.processed_wav(wav_file_path)
    print([x.class_predicted for x in res])


License
=========

Please see LICENCE.txt in this directory.
