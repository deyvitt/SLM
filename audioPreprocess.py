# Remember to Pip install : pip install nltk numpy Pillow opencv-python librosa

import librosa
import librosa.display
import numpy as np

def preprocess_audio(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path)
    # Compute spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    # Convert to log scale
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S 