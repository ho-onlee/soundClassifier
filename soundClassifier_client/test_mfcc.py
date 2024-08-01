import librosa, time
import numpy as np

audiowave = np.random.random(8000)
while True:
    tic = time.time()
    librosa.feature.mfcc(y=audiowave, sr=8000, n_mfcc=40)
    print(f"Process Time: {time.time()-tic} sec")