import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import math
import PIL
import matplotlib

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

# midi to notes map
midi_notes = [
    [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57], # low E
    [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62], # A
    [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67], # D
    [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72], # G
    [59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76], # B
    [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81], # E
]
# path to guitarset folder downloaded from official website
guitarset_path = "../GuitarSet/"

file_list = os.listdir(guitarset_path + 'annotation/')
annotation_filenames = [os.path.splitext(file)[0] for file in file_list]
print(len(annotation_filenames))
global_result = {"sections" : []}
skip_counter = 0
total_counter = 0
for m in range(len(annotation_filenames)):
    recordingName = annotation_filenames[m] #"05_Rock1-130-A_comp"

    # load annotation file (json format but different extension)cd 
    with open(guitarset_path + 'annotation/' + recordingName + '.jams', 'r') as file:
        data = json.load(file)

    audio_file = guitarset_path + 'audio_mono-mic/' + recordingName + '_mic.wav'
    audio, sr = librosa.load(audio_file, sr=44100)

    # CQT params
    hop_length = 1024
    n_bins = 96
    bins_per_octave = 12
    time_window = 0.2

    samples_per_window = int(sr * time_window)
    num_spectrograms = (len(audio) + samples_per_window - 1) // samples_per_window

    # parse annotation file
    filtered_data = []
    for entry in data['annotations']:
        if entry['annotation_metadata']['data_source'] in ['0', '1', '2', '3', '4', '5'] and entry['namespace'] == 'note_midi':
            filtered_data.append(entry['data'])

    current_time = 0
    # print(f"Possible spectrograms: {num_spectrograms}")
    # dropped_spectrograms = 0
    for i in range(num_spectrograms):
        total_counter += 1
        # skip = True
        prev_time_begin = current_time
        current_time += time_window
        # for frets
        res = [[0] * 19 for i in range(6)]
        # discover frets
        current_note = -1
        # middle_of_segment = (current_time - (time_window / 2))
        drop = False
        for entry in filtered_data:
            current_note += 1
            notes_played = len(entry)
            idx = -2
            optimal = -1
            for k in range(notes_played):
                note_end = min(entry[k]['time'] + entry[k]['duration'], current_time)
                note_begin = max(entry[k]['time'], prev_time_begin)
                captured = note_end - note_begin
                if optimal < captured:
                    optimal = captured
                    idx = k
            # case when all notes are played BEFORE the middle of analyzed sample
            # if optimal < time_window / 2 and optimal > 0:
            #     print(optimal)
            #     drop = True
            if optimal < time_window / 4:
                 res[current_note][0] = 1
            else:
                note_end = entry[idx]['time'] + entry[idx]['duration']
                # determining which note is played
                for j in range(18):
                    if(midi_notes[current_note][j] == round(entry[idx]['value'])):
                        res[current_note][j + 1] = 1
                        break
                if j == 17 and res[current_note] == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]:
                    res[current_note][j + 1] = 1
        # if skip:
        #     continue
        # if drop == True:
        #     dropped_spectrograms += 1
        #     print()
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
        section = {}
        section["strings"] = res
        
        tet = magnitude_spectrogram.flatten().tolist()
        section["spectrogram"] = tet
        global_result["sections"].append(section)
        

    print(f"files done: {m + 1}")

print(f"not skip counter: {skip_counter}")
print(f"total counter: {total_counter}")
print("checkP")
with open("global_res.json", "w") as outfile:
    json.dump(global_result, outfile, indent=2)
