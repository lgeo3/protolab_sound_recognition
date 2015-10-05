__author__ = 'lgeorge'

from resources.sound_file import SendSoundFile
import time
import os
import json


class ClassifySoundApi(SendSoundFile):
    def __init__(self, sound_classification_obj=None, api=None, **kwargs):
        super(ClassifySoundApi, self).__init__(**kwargs)
        self.sound_classification_obj = api.sound_classification_obj

    def post(self):
        #res = super(ClassifySoundApi, self).post()
        args = self.reqparse.parse_args()
        print("args is %s" % args)
        data_file = args['audio_file']
        device_type = args.pop('device_type', 'unknown')
        content_class = args.pop('content_classification', 'unknown')
        print(data_file)
        filename = self.add_entry(data=data_file, content_classification = content_class, device_type=device_type)
        #return json.dumps({'filename': filename}), 201
        args = self.reqparse.parse_args()
        print("args is {}".format(args))
        data_file = args['audio_file']

        file_with_path = os.path.join(self.saving_path, filename)
        rStart_time = time.time()

        try:
            res_classif = self.sound_classification_obj.processed_wav(file_with_path)
            prediction = '_'.join(['_'.join([str(x.class_predicted), '{0:.2f}'.format(x.confidence)]) for x in res_classif[0:10]])
            rDuration = time.time() - rStart_time
            print("duration of processing is {}".format(rDuration))
            print("RES on server is %s" % str(res_classif))
            res = json.dumps({'filename':filename, 'classif': res_classif}), 201
        except ValueError as e:
            print("value error exception %s" % e)  # TODO: fix this bug
            res = json.dumps({'exception nan value error':e}, 404)
        except Exception as e:
            print("unknown exception %s" % e)
            res = json.dumps({'exception':e}), 404
        finally:
            os.remove(file_with_path)
            return res
