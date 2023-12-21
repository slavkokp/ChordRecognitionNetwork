# ChordRecognitionNetwork

Application to create neural network capable of detecting chords as combination of strings and frets of six string guitar in standard tuning.

To train the network you have to create dataset using GuitarSet dataset (or any other one with same formatting) in PrepareDataset.py

Using main.py you can train and validate network. It is possible to use imbalanced dataset sampler but you have to provide own labels.

To analyze an audio you have to save full network using main and then use it in analyze_song.py.

