import sounddevice as sd
import numpy as np
import librosa
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

    def start_stream(self):
        try:
            with sd.InputStream(samplerate=self.samplerate, channels=1, callback=self.callback):
                print("Streaming started. Press Ctrl+C to stop.")
                while True:  # Run indefinitely
                    sd.sleep(self.duration * 1000)  # Sleep for a short duration to keep the stream alive
                    print(np.array(self.audio_data).shape)
                    threading.Thread(target=self.get_mfcc).start()  # Start get_mfcc in a new thread
        except KeyboardInterrupt:
            print("Streaming stopped by user.")

    def get_mfcc(self):
        with self.lock:  # Lock access to audio_data
            if self.audio_data:  # Check if there is audio data to process
                audio_waveform = np.concatenate(self.audio_data).flatten()
                self.audio_data = []  # Clear audio_data after processing
                mfccs = librosa.feature.mfcc(y=audio_waveform, sr=self.samplerate, n_mfcc=40)
                print("MFCCs computed.")  # Indicate that MFCCs have been computed
                return mfccs


if __name__ == "__main__":
    analyzer = AudioAnalyzer()
    analyzer.start_stream()
