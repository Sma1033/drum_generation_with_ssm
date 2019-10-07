Drum Generation with Self-Similarity Matrix (SSM)
==================

This is the supplemental repository for ISMIR 2019 paper **GENERATING STRUCTURED DRUM PATTERN USING VARIATIONAL AUTOENCODER AND SELF-SIMILARITY MATRIX**. The goal of this project is using machine learning model to generate long drum-track for a given song.
With SSM as song structure representation, we successfully demonstrate that it is possible to generate drum-track with long-term consistency. This project uses Google's [TensorFlow](https://www.tensorflow.org/ "link") and [Jupyter Notebook](https://github.com/jupyter/notebook "link") to make the implementation easier.

## The Flow of Generation
![Generation Flow](misc/generation_flow.png "Generation Flow")

- **Data Pre-Processing**: Melodic-track MIDI data is synthesized into 44.1 kHz mono-channel audio, then, converted and divided into CQT spectrogram on bar basis. Each bar is a 84 x 96 matrix. Drum tracks are down-sampled to 46 x 16 for each bar.
- **Bar Selection**: To encode song structure, 7-nearest bars are identified for every bar-level spectrogram based on drum SSM to provide relevant information for generating rhythm compatible drum patterns.
- **Training**: Selected 8-bar spectrogram are feed into a VAE-based drum pattern generator as input. Symbolic drum track data is used as ground truth to minimizing loss term.
- **Generation**: Similar procedure is performed to generate drum patterns for each single bar except inferred drum SSM is applied for bar selection. 

Listening example is available [here](https://sma1033.github.io/drum_generation_with_ssm/ "link"). (It may take a while to read audio files.)


## Environment
1. Ubuntu 16.04
2. Jupyter Notebook
3. Python 3.5+ 
4. Tensorflow 1.13+
5. Librosa
6. FluidSynth


## To Be Continued