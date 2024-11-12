import math
import os
from random import random
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
#import torchaudio
import numpy as np
import re
from scipy.io import wavfile


# import matplotlib.pyplot as plt
# import librosa.display


# This function computes LCM
def compute_lcm(x, y, gcd):
   lcm = (x*y)//gcd
   return lcm


# Function converting np read audio to range of -1 to +1
def audio_converter(audio):
    if audio.dtype == 'int16':
        return audio.astype(np.float32, order='C') / 32768.0
    else:
        print('unimplemented audio data type conversion...')
def shuffle(audio):
    return random.sample(audio, len(audio))

## sound category = [guitar, bass, vocal, speech, drums, guitar loops, bass loops, drums loops, songs, test signals, playing techniques]
class AudioDataset(Dataset):
    def __init__(self, root_dir,
                 compose_data,
                 set,
                 parameters=0,
                 composition=[],
                 fs=48000,
                 input_size=1,
                 output_size=1,
                 split=[0.8, 0.1, 0.1],
                 test_set_composition=[],
                 test_mode=[],
                 batch_size=1024,
                 data_type=torch.float32):
        """
           Initializes a data generator object
             :param root_dir: the directory in which data are stored [string]
             :param parameters: number of parameter to consider, if none is euqal to 0 [int]
             :param batch_size: the size of each batch returned by __getitem__
             :param input_size: the input size
             :param output_size: the output size
             :param fs: the sampling rate
             :param split: the train/val/test set split [vector]
             :param composition: number of minutes to include in dataset for each sound category [vector]
             :param test_set_composition: which sound category to include in test set. If None it will be mixed [vector]
             :param data_type: data type
             :param save_dataset: option to save the dataset composition
             :param test_mode: if test unseen condintioning, unseen audio or both
           """

        self.root_dir = root_dir
        self.parameters = parameters
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.fs = fs
        self.split = split
        self.composition = composition
        self.test_set_composition = test_set_composition
        self.test_mode = test_mode
        self.data_type = data_type
        self.conds = None

        # List audio files in the directory
        #self.audio_files = [f for f in os.listdir(root_dir) if f.endswith('.wav')]

        if compose_data:
            self.data = self.compose_dataset()

        self.retrieve_set(set)
        self.on_epoch_end()

    def compose_dataset_and_save(self):  #TODO: remember to consider parameter, one file per parameter combo maybe, and how we divide input and target

        #example composition=[5, 4, 3, 2, 1, 0, 2, 3, 4, 1, 3, 5]
        #example test_set_composition=[5, 4, 3, 2, 1, 0, 2, 3, 4, 1, 3, 5]*self.split[2]
        # here we concatenate the audio files based on composition, test_set_composition and split arguments
        # we save the training, validation and test set as vectors of dimension 1 (?)

        audios = []
        conds = None
        for idx, min in enumerate(self.composition):

            dir = os.path.join(self.root_dir, self.category[idx])
            audio_files = [f for f in os.listdir(root_dir) if f.endswith('.wav')] # I am assuming on unique file because easier to cut

            if self.parameters != 0:
                conds = np.zeros((len(audio_files), self.parameters))

            for n, file in enumerate(audio_files):
                filepath = os.path.join(dir, file)
                # better use a standard package to load audio, without assume torch, tf or anything else
                _, audiofile = wavfile.read(filepath)

                if conds is not None:
                    filename = os.path.split(filepath)[-1]
                    filename = filename[:-4]
                    param = filename.split('-')[1] # depend on convention
                    for i in range(self.parameters):
                        conds[n, i].append(float(param.split('_')[i]))

                if self.datatype == 'float':
                    audiofile = audio_converter(audiofile)

                if audiofile.shape[0] > 1:
                    audiofile = np.mean(audiofile, axis=0, keepdims=False) # Convert to mono

                audiofile = audiofile[:min*self.fs]

            audios.append(audiofile)

        # TODO: split input and target and split between sets using test_set_composition and test_mode argument
        # TODO: if keeping the audio slided we should have conds with similar length
        # save [audios, conds] #TODO: decide the format

        # Load audio
        #audio_path = os.path.join(self.root_dir, audio_file)
        #audio, sample_rate = torchaudio.load(audio_path)
        #audios = torch.tensor(audio, dtype=self.data_type)
        #conditions = torch.tensor(self.conds, dtype=self.data_type)

        return NotImplementedError

    def retrieve_set(self, set): #TODO: remember to consider parameter, one file per parameter combo maybe

        # here we load the desired and already saved set (argument tell which file to be loaded)
        # how we store data? h5?
        data = None
        gcd = math.gcd(self.batch_size, self.output_size)
        lcm = compute_lcm(self.batch_size, self.output_size, gcd)
        requested_length = (data.shape[0]//lcm)*lcm

        data = np.pad(data, pad_width = (0, requested_length-data.shape[0]))
        self.slided_audios = F.unfold(data, kernel_size=(self.input_size), stride=(self.output_size))

    def on_epoch_end(self): #TODO: remember to consider parameter, one file per parameter combo maybe
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(0, self.slided_audios.shape[0])

    def __len__(self, set):
        return len(self.audio_files)*int(set / self.batch_size)

    def __getitem__(self, idx):

        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        audios = self.slided_audios[indices]
        conditions = self.conds[indices]

        return audios, conditions


# def collate_fn(batch):
#     audios, conds = zip(*batch)
#
#     # Pad sequences for batch processing (?)
#     #max_audio_length = max(audio.shape[1] for audio in audios)
#     #audios = [np.pad(audio, ((0, 0), (0, max_audio_length - audio.shape[1]))) for audio in audios]
#
#     audios = torch.tensor(audios, dtype=self.data_type)
#
#     return audios, conds


# Initialize dataset and dataloader
root_dir = "../Files"
dataset = AudioDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)#, collate_fn=collate_fn)

# Sanity check
for audio, midi in dataloader:
    print("Audio:", audio.shape)
    print("MIDI:", midi.shape)