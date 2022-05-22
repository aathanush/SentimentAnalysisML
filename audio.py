#importing the modules
import librosa
import os
import time
import numpy as np
import pandas as pd

#list of emotions in tess dataset
emotions=['angry','disgust','fear','ps','happy','sad']


def extract_feature(file_name, mfcc):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((mfccs))
        return result
    else:
        return None
