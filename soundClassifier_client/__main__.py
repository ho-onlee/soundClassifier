#!/venv/bin python
import warnings
warnings.filterwarnings('ignore')
import HOS_client
import time, pickle, datetime
import sounddevice as sd
import numpy as np
import toml, os
import keras
import tensorflow as tf
import librosa

class analyzer:
    def __init__(self, model_name:str, label_name:str='label_list.pkl'):
        self.base_dir = os.path.dirname(__file__)
        self.model = self.__load_model()
        self.map = self.__load_map(os.path.join(self.base_dir, label_name))
        print("Analyser initialized")
        
    def __load_model(self):
        model = tf.keras.models.load_model(os.path.join(self.base_dir, [f for f in os.listdir(self.base_dir) if '.h5' in f and '.weights.h5' not in f][0]))
        model.load_weights(os.path.join(self.base_dir, [f for f in os.listdir(self.base_dir) if '.weights.h5' in f][0]))
        return model

    def __load_map(self, fname:str):
        with open(os.path.join(self.base_dir, fname), 'rb') as f:
            ret = pickle.loads(f.read())
        return ret
        
    def predict(self, audiowave:np.array, sr:int=48000):
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


def callback(indata, outdata, frames, time, status):
    raw_pred = anal.predict(indata)
    prediction = str(anal.map[np.argmax(raw_pred)])
    now = datetime.datetime.now()
    t = str(now.strftime('%a, %d %b %Y %H:%M:%S:%f'))
    print(f"[{t}] Prediction: {prediction}")
    with open(os.path.join(anal.base_dir, config['Output']['output_csv_fname']+'.csv'),'a+', newline='') as f:
        f.write(f"{now}, {prediction}, {raw_pred}".replace('\n', '')+"\n")
    if config['HOS_server']['HOS_available']:
        ret = node.postMessage([str(t), str(raw_pred), str(prediction)])
        print(ret)
def main():
    try:
        with sd.Stream(
                    samplerate=48000, blocksize=int(48000*0.5),
                    channels=1, callback=callback) as f:
            print('#' * 80)
            print('press Return to quit')
            print('#' * 80)
            input()
            
    except Exception as e:
        print(e)


if __name__ == '__main__':
    try:        
        config = toml.load(os.path.join(os. path.dirname(__file__), 'config.toml'))
        anal = analyzer(config['Weights']['model_name'], 
                        label_name=config['Weights']['label_name'])
        print("pre hos")
        if config['HOS_server']['HOS_available']:
            node = HOS_client.node(client=HOS_client.client(config['General']['device_name'], 
                                                client_privilege=config['General']['client_privilege']),
                                   node_name=config['General']['node_name'], 
                                   keys=['recorded_time', 'prediction_raw', 'prediction_max'],
                                   host=config['HOS_server']['server_url'],
                                   port=config['HOS_server']['port']
                                  )
        print("pre main")
        main()
    except Exception as e:
        print(e)
        exit()
