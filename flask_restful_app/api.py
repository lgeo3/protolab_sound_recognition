# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Simple restfull service providing sound recognition to remote
"""

from flask import Flask
from flask_restful import Api, reqparse, Resource
from flask_restful import fields, marshal_with
from flask.ext.sqlalchemy import SQLAlchemy
from resources.sound_file import SendSoundFile
from resources.classify_sounds import ClassifySoundApi
import argparse

from sound_classification.classification_service import SoundClassification

import os

app = Flask(__name__)
_basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+ os.path.join(_basedir, 'app.db')

db = SQLAlchemy(app)

class SoundFile(db.Model):
    __tablename__ = 'mysound'
    id = db.Column(db.INTEGER, primary_key=True)
    filename = db.Column(db.Text)
    content_classification = db.Column(db.Text)
    device_type = db.Column(db.Text)
    mimetype = db.Column(db.Text)

    def __init__(self, filename="default", content_class="default", device_type=None, mimetype=None):
        self.filename = filename
        self.content_classification = content_class
        self.device_type = device_type
        self.mimetype = mimetype

    def __repr__(self):
        return "Id: {}, content class: {}, data file: {}".format(self.id, self.content_classification, self.audio_file)

db.create_all()
api = Api(app)

api.add_resource(SendSoundFile, '/api/SendSoundFile', '/api/SendSoundFile<id>', resource_class_kwargs={'db_session':db.session})
api.add_resource(ClassifySoundApi, '/api/ClassifySoundFile', '/api/ClassifiySoundFile<id>', resource_class_kwargs={'db_session':db.session, 'api':api})

if __name__ == '__main__':
    usage = "python api.py -p '/media/dataset/*.wav'"
    parser = argparse.ArgumentParser(description="usage is %s" % usage)
    parser.add_argument('--wav-file-pattern', '-p', type=str, default="'/media/dataset/*.wav'")
    args = parser.parse_args()

    print("args is %s" % args)
    api.sound_classification_obj = SoundClassification(wav_file_pattern=args.wav_file_pattern)
    api.sound_classification_obj.learn()

    app.run(debug=True, host='0.0.0.0')