import os
from torch.utils.data import Dataset
from scipy.io import wavfile
import numpy as np

# import matplotlib.pyplot as plt


# Function converting np read audio to range of -1 to +1
def audio_converter(audio):
    if audio.dtype == 'int16':
        return audio.astype(np.float32, order='C') / 32768.0


## sound category = [guitar, bass, vocal, speech, drums, guitar loops, bass loops, drums loops, songs, test signals, playing techniques]

class AudioDatasetRecorder(Dataset):
    def __init__(self, root_dir, fs, datatype, composition=None):
        """
        Initializes a data recorder object
          :param root_dir: the directory in which data are stored [string]
          :param composition: number of minutes to include in dataset for each sound category [vector]
          :param datatype: consider the audio in float or int [str]
          :param fs: sampling rate [int]

        """

        self.root_dir = root_dir
        self.composition = composition
        self.category = None
        self.datatype = datatype

        self.impulse = np.zeros(1024)
        self.impulse[1024//2] = 1.
        self.fs = fs
        self.audios = []
    def compose_input_files(self):
        ### from the composition vector, the function retrieve the amount of requested minute from a specific sound category
        ### the audio are concatenated (with a impulse at the beginning and end of the file)
        ### for each category, a different audio file is obtained and stored

        for idx, min in enumerate(self.composition):

            audio_imp = self.impulse # we start with the impulse

            # List audio files in the directory
            dir = os.path.join(self.root_dir, self.category[idx])
            audio_files = [f for f in os.listdir(root_dir) if f.endswith('.wav')] # I am assuming on unique file because easier to cut

            audiofile = os.path.join(dir, audio_files)
            # better use a standard package to load audio, without assume torch, tf or anything else
            _, audiofile = wavfile.read(audiofile)

            if self.datatype == 'float':
                audiofile = audio_converter(audiofile)

            if audiofile.shape[0] > 1:
                audiofile = np.mean(audiofile, axis=0, keepdims=False) # Convert to mono

            audiofile = audiofile[:min*self.fs]
            audio = np.concatenate([audio_imp, audiofile])
            audio = np.concatenate([audio, audio_imp])

            self.audios.append(audio)

        return self.audios



# Initialize dataset and dataloader
root_dir = "../Files"
dataset = AudioDatasetRecorder(root_dir, fs=48000, composition=[1,1,1])
