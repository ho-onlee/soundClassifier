import tensorflow as tf
import sys
import librosa, threading, time
import sounddevice as sd
import numpy as np
import queue, pathlib
import socket
import argparse
import python_speech_features as psf
import platform
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

class tfModel:
    def __init__(self, model_path:str, labels_path:str=None):
        try:
            delegate = tf.lite.experimental.load_delegate(EDGETPU_SHARED_LIB)
        except ValueError:
            delegate = None
        
        if delegate:
            print(f"Using EdgeTPU delegate: {EDGETPU_SHARED_LIB}")
            self.interpreter = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
        else:
            print("Using CPU delegate")
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = None
        if labels_path is not None:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]

    def predict(self, mfcc_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], mfcc_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])   
        best_output = np.argmax(output_data[0])
        confidance = output_data[0][best_output]
        res = dict(out=output_data, refined=self.labels[best_output], confidance=confidance)
        return res

class keras:
    def __init__(self, model_path:str, labels_path:str=None):
        self.model = tf.keras.models.load_model(model_path)
        self.labels = None
        if labels_path is not None:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
    
    def predict(self, mfcc_data):
        output_data = self.model.predict(mfcc_data)
        best_output = np.argmax(output_data[0])
        confidance = output_data[0][best_output]
        res = dict(out=output_data, refined=self.labels[best_output], confidance=confidance)
        return res

class AudioAnalyzer:
    def __init__(self, audio_path:str=None):
        self.audio_queue = queue.Queue()
        self.model = None
        self.processingTime = []
        self.enable_thread = False

    def load_model(self, modelType:str='tflite', model_path:str=None, labels_path:str=None):
        if modelType == "tflite":
            self.model = tfModel(model_path, labels_path)
        elif modelType == "keras":
            self.model = keras(model_path, labels_path)

    def process_audio(self, audio_data, sample_rate):
        def audio_to_mfcc(audio_data, sample_rate):
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40, n_fft=512, center=True)
            print(mfccs.shape)
            return mfccs
        # Extract MFCC features
        tic = time.time()
        print(f"Process starts")
        mfccs = audio_to_mfcc(audio_data, sample_rate)
        tac = time.time()   
        print(f"\tMFCC conversion: {tac-tic}sec")
        mfcc_scaled = np.mean(mfccs.T, axis=2)
        print(mfcc_scaled.shape)
        input_data = np.reshape(mfcc_scaled, (1, 40, 1))
        print(f"\tinput data preparation: {time.time()-tac}sec")
        prediction = self.model.predict(input_data)
        tok = time.time()
        print(f"\tProcessing Time: {tok-tic}sec")
        self.processingTime.append([tic, tac, tok, tac-tic, tok-tac, tok-tic])

    def streamCallback(self, indata, frames, time, status):
        self.audio_queue.put(indata.copy())

    

    def start_processing(self, sample_rate):
        while True:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                if self.enable_thread:
                    threading.Thread(target=self.process_audio, args=(audio_data, sample_rate)).start()
                else:
                    self.process_audio(audio_data, sample_rate)
            time.sleep(0.01)

def audio_streamer(chunks, duration, sample_rate):
    # print(f"Processing {len(chunks)} chunks of audio data")
    for chunk in chunks:
        analyzer.streamCallback(chunk, sample_rate, 0, 1)
        time.sleep(duration)

if __name__ == "__main__":
    duration = 3
    analyzer = AudioAnalyzer()
    parser = argparse.ArgumentParser(description='Audio processing with model type')
    parser.add_argument('--model_type', type=str, required=True, choices=['keras', 'tflite'], help='Type of model to use (keras or tflite)')
    parser.add_argument('--duration', type=int, default=1, help='Duration of each audio chunk in seconds')
    args = parser.parse_args()

    model_type = args.model_type
    if model_type == "tflite":
        analyzer.load_model('tflite', model_path="../soundClassifier/model.tflite", labels_path="../soundClassifier/labels.txt")
    elif model_type == "keras":
        analyzer.load_model('keras', model_path="../soundClassifier/my_model.keras", labels_path="../soundClassifier/labels.txt")
    audio_path = "example.mp3"
    audio_data, sample_rate = librosa.load(audio_path, sr=24000)
    
    chunk_size = sample_rate * duration
    chunks = np.array_split(np.reshape(audio_data, (len(audio_data),1)), len(audio_data) // chunk_size)

    
    audio_thread = threading.Thread(target=audio_streamer, args=(chunks, duration, sample_rate), daemon=True)
    audio_thread.start()
    # Start the audio stream and processing thread
    processing_thread = threading.Thread(target=analyzer.start_processing, args=(sample_rate,), daemon=True).start()
    audio_thread.join()
    while True: 
        if analyzer.audio_queue.empty():
            break
        
    with open(f"processing_time_{socket.gethostname()}.csv", 'w') as f:
        f.write("Processing Time\n")
        for t in analyzer.processingTime:
            t = ",".join(map(str, t))
            f.write(f"{t}\n")
            
    dur = [a[-1]for a in analyzer.processingTime]
    print(f"Processing time: {np.mean(dur)} : {np.std(dur)}")
