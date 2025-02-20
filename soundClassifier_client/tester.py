from executer_indi import *
import os
import urllib.parse
import librosa

def load_audio_files(file_path):
    """Loads audio files from the specified directory."""
    audio_data, samplerate = librosa.load(file_path, sr=None)  # Load audio file with original sampling rate
    return audio_data, samplerate


if __name__ == "__main__":
    encoded_str = "../soundClassifier/dataset/d95e05c6-%EA%B8%88%EC%9A%94%EC%9D%BC_%EC%98%A4%ED%9B%84_1-26.m4a"
    decoded_str = urllib.parse.unquote(encoded_str)
    audio_files, sr = load_audio_files(decoded_str) # Load audio files from the dataset directory
    chunk_size = 1 * sr  # Define chunk size as 3 seconds
    hop_size=int(len(audio_files)/100)
    audio_chunks = [audio_files[i:i + chunk_size] for i in range(0, len(audio_files) - chunk_size + 1, hop_size)]  # Split audio_files into chunks with hops
        # Assuming there's a method in the AudioAnalyzer class to handle audio data
    print(len(audio_chunks))
    analyzer = AudioAnalyzer()
    data = []
    for idx, chunck in enumerate(audio_chunks):
        analyzer.audio_waveform = chunck  # Set the audio waveform
        analyzer.samplerate = sr  # Set the samplerate
        ret = analyzer.process_audio()  # Call the process_audio method to handle the audio data
        data.append(ret)
        # output_dir = 'new_dataset'
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        
        # output_file_path = os.path.join(output_dir, f'output_{idx}_{ret["prediction_text"]}.wav')
        # sf.write(output_file_path, ret['audio_waveform'], sr)  # Save the audio waveform to a new file

    import matplotlib.pyplot as plt
    
    # Plot audio spectrum
    plt.figure(figsize=(10, 6))
    
    # Plot the audio spectrum on the top
    plt.subplot(2, 1, 1)
    plt.title("Audio Spectrum")
    plt.specgram(audio_files, NFFT=1024, Fs=sr, Fc=0, noverlap=512, cmap='plasma', sides='default', mode='default')
    plt.colorbar(label='Intensity (dB)')
    plt.ylabel('Frequency (Hz)')
    
    # Plot the gradient of each output label to ret['prediction_raw'] on the bottom
    plt.subplot(2, 1, 2)
    plt.title("Label Predictions Gradient")
    for d in ret['pred_raw']:
        plt.plot(d, '*')    
        
    plt.tight_layout()
    plt.show()

