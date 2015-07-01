# -*- coding: utf-8 -*-
#!/usr/bin/env python

from werkzeug.datastructures import FileStorage
from flask_restful import fields, marshal_with, reqparse, Resource
import hashlib
import json
import os

class SendSoundFile(Resource):
    def __init__(self, db_session=None):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument('audio_file', type=FileStorage, location='files')
        self.reqparse.add_argument('content_classification', type=str, location='form')
        self.reqparse.add_argument('device_type', type=str, location='form')

        self.db_session = db_session
        super(SendSoundFile, self).__init__()
        self.saving_path="./data"

    def add_entry(self, data=None, content_classification='', device_type=None):
        print("content_clas si : {}".format(content_classification))
        sig = hashlib.sha1()
        for line in data.stream.read():
            sig.update(line)
        filename = sig.hexdigest()

        from api import SoundFile
        s = SoundFile(filename=filename, content_class=content_classification, mimetype=data.mimetype, device_type=device_type)
        self.db_session.add(s)
        self.db_session.commit()
        self.db_session.flush()

        # saving file on hdd
        data.stream.seek(0)
        file = os.path.join(self.saving_path, filename)
        data.save(file)
        print("Saving file at {}".format(file))
        return filename

    def get(self, id):
        return 201
        pass

    def post(self):
        """
        Create a new entry with a new file
        :return:
        """
        args = self.reqparse.parse_args()
        print("args is %s" % args)
        data_file = args['audio_file']
        device_type = args.pop('device_type', 'unknown')
        content_class = args.pop('content_classification', 'unknown')
        filename = self.add_entry(data=data_file, content_classification = content_class, device_type=device_type)
        return json.dumps({'filename': filename}), 201



