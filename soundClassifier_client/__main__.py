#!.venv/bin python
import warnings
warnings.filterwarnings('ignore')
import HOS_client
import time, pickle, datetime
import sounddevice as sd
import numpy as np
import toml, os
import keras, threading
import tensorflow as tf
import librosa, h5py, faulthandler
from librosa.feature import mfcc
import numba
from numba import jit

from numba import int32, float32    # import the types
from numba.experimental import jitclass

# faulthandler.enable()
spec = [
    ('model_name', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

@jitclass(spec)
class analyzer(object):
    def __init__(self, model_name:str):
        self.base_dir = os.path.dirname(__file__)
        self.model = self.__load_model()
        
    def __load_model(self):
        model, self.map = self.__load_model_ext(os.path.join(self.base_dir, [f for f in os.listdir(self.base_dir) if '.h5' in f and '.weights.h5' not in f][0]))
        model.load_weights(os.path.join(self.base_dir, [f for f in os.listdir(self.base_dir) if '.weights.h5' in f][0]))
        return model
        
    def __load_model_ext(self, filepath:str, custom_objects=None):
        model = tf.keras.models.load_model(filepath, custom_objects=None)
        f = h5py.File(filepath, mode='r')
        meta_data = None
        if 'label_data' in f.attrs:
            label_data = f.attrs.get('label_data')
        f.close() 
        return model, label_data
        
    def predict(self, audiowave:np.array, sr:int=16000):
        prediction = np.zeros((1, len(self.map)))
        c = 0
        for i in range(int(np.floor(len(audiowave)/(0.45*sr)))+1):
            c+=1
            start = int(i*(0.45*sr))
            end = int(start+(0.92*sr))
            audio_94 = audiowave[start:end]
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_94, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfccs.T, axis=2)
            input_data = np.reshape(mfcc_scaled, (1, 40))
            prediction += self.model(input_data)
        return prediction.numpy()/c

    @jit(parallel=True, nopython=True)
    def predict2(self, audiowave:np.array, sr:int=16000):
        tic = time.time()
        prediction = np.zeros((1, len(self.map)))
        # Extract MFCC features
        mfccs = mfcc(y=audiowave, sr=sr, n_mfcc=40)
        print(f"MFCC conversion: {time.time()-tic}sec")
        mfcc_scaled = np.mean(mfccs.T, axis=2)
        input_data = np.reshape(mfcc_scaled, (1, 40))
        print(f"input data preparation: {time.time()-tic}sec")
        prediction += self.model(input_data)
        print(f"Model Prediction: {time.time()-tic}sec")
        return prediction.numpy()

@jit(parallel=True, nopython=True)
def process(indata):
    raw_pred = anal.predict2(indata, config['Audio_Setting']['sample_rate'])
    prediction = str(anal.map[np.argmax(raw_pred)])
    now = datetime.datetime.now()
    t = str(now.strftime('%a, %d %b %Y %H:%M:%S:%f'))
    ref = 1 
    
    S = np.abs(librosa.stft(y=indata))
    dbp = librosa.power_to_db(S**2, ref=ref).mean()

    rms = librosa.feature.rms(y=indata)
    dbs = librosa.amplitude_to_db(rms, ref=ref)
    A_weighting = librosa.A_weighting(dbs)
    dBA = librosa.amplitude_to_db(rms * A_weighting, ref=ref)[0][0][0]
    
    print(f"[{t}] Prediction: {prediction}")
    writeCSV(now, prediction, raw_pred, dbp, dBA)


def writeCSV(now, prediction, raw_pred, dbp, dBA):
    if config['Output']['output_csv_fname']+'.csv' not in os.listdir(anal.base_dir):
        with open(os.path.join(anal.base_dir, config['Output']['output_csv_fname']+'.csv'),'w', newline='') as f:
            f.write("Time, Prediction, raw_pred, dBFS[power], dBs, dBA".replace('\n', '')+"\n")
    with open(os.path.join(anal.base_dir, config['Output']['output_csv_fname']+'.csv'),'a+', newline='') as f:
        f.write(f"{now}, {prediction}, {raw_pred}, {dbp}, {dbs.mean()}, {dBA}".replace('\n', '')+"\n")
    if config['HOS_server']['HOS_available']:
        ret = node.postMessage([str(t), str(raw_pred), str(prediction), str(dbp), str(dbs.mean()), str(dBA)])
        print(ret)
        
def callback(indata, outdata, frames, time, status):
    if config['General']['multi-threading']:
        if threading.active_count() < config['General']['max_thread']:
            tr = threading.Thread(target=process, args=(indata,))
            tr.start()
            print(f"{tr.getName()} Started; {threading.active_count()}/{config['General']['max_thread']} alive")
    else:
        process(indata)
    
        
def main():
    try:
        if config['Audio_Setting']['device'] == 'none':
            with sd.Stream( 
                        samplerate=config['Audio_Setting']['sample_rate'], 
                        blocksize=int(config['Audio_Setting']['sample_rate'] * config['Audio_Setting']['duration']),
                        channels=1, callback=callback) as f:
                print('#' * 80)
                print('press Return to quit')
                print('#' * 80)
                input()
        else:
            with sd.Stream( device = config['Audio_Setting']['device'],
                        samplerate=config['Audio_Setting']['sample_rate'], 
                        blocksize=int(config['Audio_Setting']['sample_rate'] * config['Audio_Setting']['duration']),
                        channels=1, callback=callback) as f:
                print('#' * 80)
                print('press Return to quit')
                print('#' * 80)
                input()
            
            with sd.Stream( 
                        samplerate=config['Audio_Setting']['sample_rate'], 
                        blocksize=int(config['Audio_Setting']['sample_rate'] * config['Audio_Setting']['duration']),
                        channels=1, callback=callback) as f:
                print('#' * 80)
                print('press Return to quit')
                print('#' * 80)
                input()
            
    except Exception as e:
        print(e)


if __name__ == '__main__':
    try:        
        print(sd.query_devices())
        config = toml.load(os.path.join(os. path.dirname(__file__), 'config.toml'))
        anal = analyzer(config['Weights']['model_name'])
        if config['HOS_server']['HOS_available']:
            node = HOS_client.node(client=HOS_client.client(config['General']['device_name'], 
                                                client_privilege=config['General']['client_privilege']),
                                   node_name=config['General']['node_name'], 
                                   keys=['recorded_time', 'prediction_raw', 'prediction_max', 'dBFS_Power', 'dBs_Amp', 'dBA_Amp'],
                                   host=config['HOS_server']['server_url'],
                                   port=config['HOS_server']['port']
                                  )
        main()
        pass
    except Exception as e:
        print(e)
        exit()
