import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import threading  # Import threading module

class AudioAnalyzer:
    def __init__(self, samplerate=16000, duration=5):
        self.samplerate = samplerate
        self.duration = duration
        self.audio_data = []
        self.lock = threading.Lock()  # Create a lock for thread safety

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        with self.lock:  # Lock access to audio_data
            self.audio_data += list(indata.copy())

    def audio_stream(self):
        with sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.callback):
            print("Audio streaming started. Press Ctrl+C to stop.")
            while True:  # Run indefinitely
                sd.sleep(self.duration * 1000)  # Sleep for a short duration to keep the stream alive

    def start_stream(self):
        try:
            # Start the audio stream in a separate thread
            threading.Thread(target=self.audio_stream, daemon=True).start()
            while True:  # Main thread can perform other tasks
                print(np.array(self.audio_data).shape)
                threading.Thread(target=self.process_audio).start()  # Start get_mfcc in a new thread
                sd.sleep(self.duration * 1000)  # Sleep to control the frequency of MFCC processing
        except KeyboardInterrupt:
            print("Streaming stopped by user.")
            
    def process_audio(self):
        mfcc = self.get_mfcc()
        if mfcc:
            self.predict_labels(mfcc, model_path='../soundClassifier/model.tflite')

    def get_mfcc(self):
        with self.lock:  # Lock access to audio_data
            if self.audio_data:  # Check if there is audio data to process
                audio_waveform = np.concatenate(self.audio_data.copy()).flatten()
                self.audio_data = []  # Clear audio_data after processing
            else:
                return None
        mfccs = librosa.feature.mfcc(y=audio_waveform, sr=self.samplerate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        print("MFCCs computed.")  # Indicate that MFCCs have been computed
        return mfccs_scaled
            
    def predict_labels(self, x_test, model_path='model.tflite'):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x_test_input = np.array([x_test.reshape(1, len(x_test), 1)], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], x_test_input[0])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_classes_tflite = np.argmax(output_data, axis=1)[0]        
        return predicted_classes_tflite

if __name__ == "__main__":
    analyzer = AudioAnalyzer()
    analyzer.start_stream()
