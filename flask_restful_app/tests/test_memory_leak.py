import requests
wav_file = "example.wav"
server_ip = "127.0.0.1"
webservice_pageaddress = "http://%s:5000/api/ClassifySoundFile" % server_ip

wav_file = "/tmp/bell.wav"
device_type = ''


for i in range(100):
	with open(wav_file, 'r') as audio_file:
		payload = {'content_classification': 'unknown', 'device_type':device_type}
		files = {'audio_file':audio_file}
		print(files)
		r = requests.post(webservice_pageaddress, data=payload, files=files)


