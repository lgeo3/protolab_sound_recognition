__author__ = 'lgeorge'
from setuptools import setup, find_packages


setup(name='protolab_sound_recognition',
      version='0.0.1',
      description='sound recognition in python',
      author='laurent george',
      author_email='lgeorge@aldebaran-robotics.com',
      license='MIT',
      packages=find_packages(),
      #packages=['sound_processing', 'sound_classification', 'flask_restful_app', 'flask_restful_app.common'],
      zip_safe=False)
