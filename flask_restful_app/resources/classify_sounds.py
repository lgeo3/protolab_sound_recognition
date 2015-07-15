__author__ = 'lgeorge'

from sound_file import SendSoundFile
import time
import os
import json


class ClassifySoundApi(SendSoundFile):
    def __init__(self, sound_classification_obj=None, api=None, **kwargs):
        super(ClassifySoundApi, self).__init__(**kwargs)
        if api.sound_classification_obj is None:
            print("error api is None")
        self.sound_classification_obj = api.sound_classification_obj


    def post(self):
        #res = super(ClassifySoundApi, self).post()
        args = self.reqparse.parse_args()
        print("args is %s" % args)
        data_file = args['audio_file']
        device_type = args.pop('device_type', 'unknown')
        content_class = args.pop('content_classification', 'unknown')
        filename = self.add_entry(data=data_file, content_classification = content_class, device_type=device_type)
        #return json.dumps({'filename': filename}), 201
        args = self.reqparse.parse_args()
        print("args is {}".format(args))
        data_file = args['audio_file']

        file_with_path = os.path.join(self.saving_path, filename)
        rStart_time = time.time()
        res_classif = self.sound_classification_obj.processed_wav(file_with_path)
        prediction = '_'.join(['_'.join([str(x.class_predicted), '{0:.2f}'.format(x.confidence)]) for x in res_classif[0:10]])
        os.rename(file_with_path, file_with_path.replace('.wav', '_{}.wav'.format(prediction)))  # adding prediction to filename
        rDuration = time.time() - rStart_time
        print("duration of processing is {}".format(rDuration))
        print("RES on server is %s" % str(res_classif))
        return json.dumps({'filename':filename, 'classif': res_classif}), 201


