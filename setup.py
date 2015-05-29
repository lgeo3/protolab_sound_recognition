__author__ = 'lgeorge'
from setuptools import setup


setup(name='protolab_sound_recognition',
      version='0.0.1',
      description='sound recognition in python',
      author='laurent george',
      author_email='lgeorge@aldebaran-robotics.com',
      license='MIT',
      packages=['sound_processing', 'sound_classification'],
      zip_safe=False,
      install_requires=['scikits.audiolab',
                        'scikit-learn',
			'pandas',
			'speech-features'
])
