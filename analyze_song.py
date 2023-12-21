import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import groupby 
import json
import math
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)
def classes_to_frets(string_class):
    if string_class == 0:
        return '-'
    return str(string_class - 1)

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

# audio path
audio_file = 'hurt.wav'

# saved "Full network" (not just dict) path
net = torch.load("../my_data/PauseChampFullResnet18layers3.json")
net.to(device)
audio, sr = librosa.load(audio_file, sr=44100)
# CQT params
hop_length = 1024
n_bins = 96
bins_per_octave = 12
time_window = 0.2


samples_per_window = int(sr * time_window)
num_spectrograms = (len(audio) + samples_per_window - 1) // samples_per_window
net.eval()
print(num_spectrograms)

spectrograms = []
for i in range(num_spectrograms):
    start_sample = i * samples_per_window
    end_sample = start_sample + samples_per_window
    audio_segment = audio[start_sample:end_sample]

    # create spectrogram
    cqt = librosa.cqt(audio_segment, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave)
    magnitude_spectrogram = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    magnitude_spectrogram = scale_minmax(magnitude_spectrogram, 0, 255)
    magnitude_spectrogram = np.flip(magnitude_spectrogram, axis=0)
    magnitude_spectrogram = 255-magnitude_spectrogram
    magnitude_spectrogram = np.round(magnitude_spectrogram, 0).astype(int)
    if magnitude_spectrogram.shape != (96, 9):
        print(i * time_window)
        continue
    magnitude_spectrogram = np.repeat(magnitude_spectrogram.reshape(1, 96, 9), 3, 0)

    spectrograms.append(magnitude_spectrogram)


res = []
with torch.no_grad():
    for i in range(len(spectrograms)):
        spectrogram = torch.div(torch.reshape(torch.FloatTensor(spectrograms[i]), (1, 3, 96, 9)), 255).to(device)
        pred = net(spectrogram)
        #print(f"time: {i * time_window}")
        predicted = [None] * 6
        for j in range(6):
            predicted[j] = pred[j].argmax(1).item()
            #print(f'String {j}: "{predicted[i]}"')
        #print()
        res.append((i * time_window, predicted))

final_res = [res[0]]
curr_time = final_res[0][0]
for i in range(1, len(res)):
    if res[i][1] == final_res[-1][1]:
        continue
    else:
        final_res.append(res[i])

entries_in_line = 10
entries = len(final_res)
blocks = math.ceil(entries / entries_in_line)
print_strings = [["" for _ in range(7)] for _ in range(blocks)]
string_label = ["t", "e", "A", "D", "G", "B", "E"]
for i in range(blocks):
    print_strings[i][0] += string_label[0] + '\t'
    for j in range(entries_in_line):
        idx = i * entries_in_line + j
        if idx < entries:
            print_strings[i][0] += str(round(final_res[i * entries_in_line + j][0], 1)) + '\t'
    print_strings[i][0] += '\n'
    for m in range(1, 7):
        print_strings[i][m] += string_label[m] + '\t'
        for j in range(entries_in_line):
            idx = i * entries_in_line + j
            if idx < entries:
                print_strings[i][m] += classes_to_frets(final_res[i * entries_in_line + j][1][m - 1]) + '\t'
        print_strings[i][m] += '\n'

with open(f"pseudo_tabs_new.txt", "w") as outfile:
    for i in range(blocks):
        outfile.write(print_strings[i][0])
        for j in reversed(range(1, 7)):
            outfile.write(print_strings[i][j])

