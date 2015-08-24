import os
import subprocess

def _get_training_data(dataset_url=None):
    """
    Download training data from public dropbox
    :return: path of dataset
    """
    if dataset_url is None:
        dataset_url = "https://www.dropbox.com/s/ekldjq8o1wfhcq1/dataset_aldebaran_6sounds.tar.gz?dl=0"
    dataset_filename = os.path.join(os.path.abspath('.'), dataset_url.split('/')[-1])
    if not(os.path.isfile(dataset_filename)):
        p = subprocess.Popen(['wget', dataset_url, '-O', dataset_filename])  # using wget simpler than urllib with droppox changing urlname in http response
        p.wait()
    dataset_path = dataset_filename + '_directory'
    p = subprocess.Popen(['mkdir', '-p', dataset_path])
    p.wait()
    command = ['tar', '-xvzf', dataset_filename, '-C', dataset_path, '--strip-components=1']
    proc = subprocess.Popen(command,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    proc.wait()
    return os.path.abspath(dataset_path)
