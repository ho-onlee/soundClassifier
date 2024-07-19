import sounddevice as sd
import numpy as np
import librosa


def callback(*args):
    indata = args[0]
    ref = 1
    # ref = 0.00000000001
    # ref=114/22
    S = np.abs(librosa.stft(y=indata))
    dbp = librosa.power_to_db(S**2, ref=ref).mean()
    
    rms = librosa.feature.rms(y=indata)
    dbs = librosa.amplitude_to_db(rms, ref=ref)
    A_weighting = librosa.A_weighting(dbs)
    dBA = librosa.amplitude_to_db(rms * A_weighting, ref=ref)[0][0][0]
    
    print(dbp, dbs, dBA)

try:
    print(sd.query_devices())
    with sd.Stream(device=(7,7),samplerate=48000, blocksize=int(48000*0.05),
                channels=1, callback=callback) as f:
        print('#' * 80)
        print(f'{f.device}press Return to quit')
        print('#' * 80)
        input()
    
except Exception as e:
    print(e)