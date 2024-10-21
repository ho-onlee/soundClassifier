import sounddevice as sd
import numpy as np
import librosa, os
import tensorflow as tf
import threading  # Import threading module
import soundfile as sf 
os.chdir(os.path.dirname(__file__))  # Change working directory to the directory of this file
# Majority voting from windows. 
# decibel cutoffs/Thresholding
#
class AudioAnalyzer:
    def __init__(self, duration=3):
        self.samplerate = 16000
        self.duration = duration
        self.audio_data = []
        self.lock = threading.Lock()  # Create a lock for thread safety
        self.audio_waveform = None
        self.prep_NN()
        with open('../soundClassifier/labels.txt', 'r') as file:
            self.labels = [line.strip() for line in file.readlines()]

    def prep_NN(self):
        """Prepares the neural network by loading the model and allocating tensors."""
        model_path='../soundClassifier/model.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def callback(self, indata, frames, time, status):
        """Callback function to handle audio input data."""
        if status:
            print(status)
        with self.lock:  # Lock access to audio_data
            self.audio_data += list(indata.copy())

    def audio_stream(self):
        """Starts the audio stream and keeps it alive indefinitely."""
        with sd.InputStream(device=1, samplerate=self.samplerate, channels=1, callback=self.callback):
            print("Audio streaming started. Press Ctrl+C to stop.")
            while True:  # Run indefinitely
                sd.sleep(self.duration * 1000)  # Sleep for a short duration to keep the stream alive

    def start_stream(self):
        """Starts the audio stream in a separate thread and processes audio data."""
        try:
            # Start the audio stream in a separate thread
            threading.Thread(target=self.audio_stream, daemon=True).start()
            while True:  # Main thread can perform other tasks
                threading.Thread(target=self.process_audio).start()  # Start get_mfcc in a new thread
                sd.sleep(self.duration * 1000)  # Sleep to control the frequency of MFCC processing
        except KeyboardInterrupt:
            print("Streaming stopped by user.")

    def process_audio(self):        
        """Processes the audio data to calculate decibels and MFCC, and predicts labels."""
        with self.lock:  # Lock access to audio_data
            if self.audio_data:  # Check if there is audio data to process
                self.audio_waveform = np.concatenate(self.audio_data.copy()).flatten()
                self.audio_data = []  # Clear audio_data after processing
        if self.audio_waveform is not None:
            db = self.calculate_decibel(self.audio_waveform)
            mfcc = self.get_mfcc(self.audio_waveform)
            if mfcc is not None:
                output_data = np.mean([self.predict_labels(mf) for mf in mfcc], axis=1)[0]
                ret = [(i, value) for i, value in enumerate(output_data) if value > 0.5]
                ret_text = ", ".join([f"{self.labels[t]}({q})" for t, q in ret ])
                return dict(audio_waveform = self.audio_waveform, samplerate = self.samplerate, prediction=ret, prediction_text=ret_text, fsdb=db) 
                # if not os.path.exists('dataset'):
                #     os.makedirs('dataset')
                # sf.write(f'dataset/{ret}.wav', self.audio_waveform, self.samplerate)

    def calculate_decibel(self, audio_waveform:list)-> float:
        """Calculates the decibel level of the given audio waveform."""
        rms = np.sqrt(np.mean(np.square(audio_waveform)))
        if rms > 0:
            decibels = 20 * np.log10(rms)
        else:
            decibels = -np.inf 
        return decibels
    
    def get_mfcc(self, audio_waveform):
        """Extracts MFCC features from the audio waveform."""
        mfccs = librosa.feature.mfcc(y=audio_waveform, sr=self.samplerate, n_mfcc=40)
        return mfccs.T

    def predict_labels(self, x_test):
        """Predicts labels for the given input using the neural network."""
        x_test_input = np.array([x_test.reshape(1, len(x_test), 1)], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], x_test_input[0])
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])  
        return output_data

if __name__ == "__main__":
    analyzer = AudioAnalyzer()
    analyzer.start_stream()
