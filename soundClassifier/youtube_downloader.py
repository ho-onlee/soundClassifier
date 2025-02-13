import pytube

def download_audio(url, start_time, end_time, output_path):
    """Download audio from a YouTube video starting at a specific second."""
    try:
        yt = pytube.YouTube(f'https://www.youtube.com/watch?v{url}&t={start_time}')
        audio_stream = yt.streams.get_audio_only()
        # audio_file = audio_stream.download(output_path)
        
        # Trim the audio file to start from the specified second
        print(audio_stream)
        
        # print(f"Downloaded and trimmed audio to: {trimmed_audio_file}")
        # return trimmed_audio_file
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

download_audio('=-Mu1AWT_x54', 300, 330, 'dataset')
