import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import csv
import io
import librosa
import IPython
import pickle
import os
import json
from paramiko import SSHClient
from scp import SCPClient
import scipy.signal as sg
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import seaborn as sns
from urllib.parse import unquote
# from scipy.ndimage.filters import gaussian_filter1d
import keras
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, plot_model
from tqdm import tqdm
from urllib.parse import unquote
from object_lib import *

def log(func):
    def log_action(*args, **kwargs):
        print(f'[{func.__name__}] ran')
        ret = func(*args, **kwargs)
        if ret is not None:
            print(f'[{func.__name__}] returned {ret}')
        else:
            print(f'[{func.__name__}] ended without return')
    return log_action

class audio_sample:
    def __init__(self, filename):
        self.filename=filename
        self.audiowave, self.sr = librosa.load(filename, sr=16000)
        # self.audiowave = librosa.effects.trim(self.audiowave)

    def stream(self):
        return librosa.stream(self.filename,
                  block_length=int(len(self.audiowave)/1000),
                  frame_length=4000,
                  hop_length=2000)
    @log
    def display(self):
        return IPython.display.Audio(self.audiowave, rate=self.sr)

    def split(self, start, end):
        return self.audiowave[int(start*self.sr):int(end*self.sr)]
    
    def plot(self):
        return plt.plot(self.audiowave)
        
    def segmentation(self, nbins:list, window_size=1):
        '''Output is called chunk'''
        window_size = int(window_size*self.sr) 
        lens = len(self.audiowave)
        assert(lens>nbins)
        assert(lens>window_size)
        steps = int(np.ceil(lens/nbins)-1 )
        i = 0
        bucket = []
        for n in range(nbins):
            loc = steps*n
            bucket.append([self.audiowave[loc:loc+window_size], loc, loc+window_size]) 
            i+= 1
        return bucket

class soundClassifier:
    def __init__(self, saved_model = 'Yamnet.sv'):
        if os.path.exists(saved_model):
            self.model = tf.saved_model.load(saved_model)
            print(f"[log]\t Model {saved_model} loaded!")
        else:
            self.model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        self.class_names = self.__class_names_from_csv(tf.io.read_file(class_map_path).numpy().decode('utf-8'))

    def __class_names_from_csv(self, class_map_csv_text):
        """Returns list of class names corresponding to score vector."""
        class_map_csv = io.StringIO(class_map_csv_text)
        class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
        class_names = class_names[1:]  # Skip CSV header
        return class_names
        
    def predict(self, audio_sample, top_n = 3, minimum_confidance=0.2):
        scores, embeddings, log_mel_spectrogram = self.model(audio_sample)
        scores.shape.assert_is_compatible_with([None, 521])
        scores = scores.numpy().mean(axis=0)
        classes = sorted([[self.class_names[ix], scores[ix]] for ix in np.argpartition(scores,-top_n)[-top_n:]], key=lambda x:x[1])
        select_only_most_likely = list(filter(lambda x:classes[-1][1] - x[1] <= 0.15  ,classes))
        with_minimum_confidance = list(filter(lambda x:x[1] >= minimum_confidance  ,select_only_most_likely))
        # [chunk, with_minimum_confidance, self.convert_to_decibel(audio)]
        return dict(audio_sample = audio_sample, 
                   select_only_most_likely = select_only_most_likely,
                   with_minimum_confidance = with_minimum_confidance,
                   embeddings = embeddings,
                   log_mel = log_mel_spectrogram)
        
    def convert_to_decibel(self, arr):
        return np.mean(librosa.power_to_db((np.abs(librosa.stft(arr)))**2, ref=1), axis=1)
        

    def segment_analyse(self, segments):
        db = [self.analyse(s) for s in segments]
        
            
    def analyse(self, chunk):
        return self.predict(chunk, top_n=1)

    def prepareData(self, csv):
        db = pd.read_csv(csv)
        for idx, a in enumerate(db.audio):
            db.audio[idx] = unquote(a.split('/')[-1])
        self.training_data = list()
        self.labelBucket = []
        for d in db.index:
            dataset = db.loc[d]
            labels = dataset.labels
            if pd.isna(labels):
                continue
            labels = json.loads(labels)
            audio = audio_sample(os.path.join('dataset', dataset.audio))
            for label in labels:
                x = audio.split(label['start'], label['end'])
                y = label['labels'][0]
                if y in self.labelBucket:
                    map = self.labelBucket.index(y)  
                else: 
                    map = len(self.labelBucket) 
                    self.labelBucket.append(y)
                # if y=='Alaris' : map = 0
                # if y=='Baxter' : map = 1 
                # if y=='ICU Medical': map = 2 
                fold = 1 if y == 'Coposition' else 0        
                predictions = self.predict(x)
                embeddings = predictions['embeddings']
                yamresult = predictions['select_only_most_likely']
                nbeddings = tf.shape(embeddings)[0]
                self.training_data.append(dict(audio_name = dataset.audio ,x=x,y=y,map=map, length_x = len(x), yamresult=yamresult, fold=fold, embeddings=embeddings.numpy().mean(axis=0)))
            

        