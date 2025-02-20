import tensorflow as tf
import sys
import librosa, threading, time
import sounddevice as sd
import numpy as np
import queue, pathlib

class tfModel:
    def __init__(self, model_path:str, labels_path:str=None):
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

class AudioAnalyzer:
    def __init__(self, audio_path:str=None):
        self.audio_queue = queue.Queue()
        self.model = None
        self.processingTime = []
        self.enable_thread = False

    def load_model(self, model_path:str=None, labels_path:str=None):
        self.model = tfModel(model_path, labels_path)

    def process_audio(self, audio_data, sample_rate):
        def audio_to_mfcc(audio_data, sample_rate):
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
            return mfccs
        # Extract MFCC features
        tic = time.time()
        mfccs = audio_to_mfcc(audio_data, sample_rate)
        mfcc_scaled = np.mean(mfccs.T, axis=2)
        input_data = np.reshape(mfcc_scaled, (1, 40, 1))
        prediction = self.model.predict(input_data)
        tok = time.time()
        self.processingTime.append(tok-tic)
        # print(f"Prediction: {prediction['refined']} with {prediction['confidance']*100}% confidence")

    def streamCallback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        # print(f"queue size: {self.audio_queue.qsize()}")
        self.audio_queue.put(indata.copy())

    

    def start_processing(self, sample_rate):
        while True:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                if self.enable_thread:
                    threading.Thread(target=self.process_audio, args=(audio_data, sample_rate)).start()
                else:
                    self.process_audio(audio_data, sample_rate)
            time.sleep(0.1)

def audio_streamer(chunks, duration, sample_rate):
    # print(f"Processing {len(chunks)} chunks of audio data")
    for chunk in chunks:
        analyzer.streamCallback(chunk, sample_rate, 0, 1)
        time.sleep(duration)

if __name__ == "__main__":
    duration = 1
    analyzer = AudioAnalyzer()
    analyzer.load_model(model_path="../soundClassifier/model.tflite", labels_path="../soundClassifier/labels.txt")
    audio_path = "example.mp3"
    audio_data, sample_rate = librosa.load(audio_path, sr=32000)
    
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
    print(f"Processing time: {np.mean(analyzer.processingTime)} : {np.std(analyzer.processingTime)}")

