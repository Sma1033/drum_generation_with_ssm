Drum Generation with Self-Similarity Matrix (SSM)
==================

This is the supplemental repository for ISMIR 2019 paper **GENERATING STRUCTURED DRUM PATTERN USING VARIATIONAL AUTOENCODER AND SELF-SIMILARITY MATRIX**. The goal of this project is using machine learning model to generate long drum-track for a given song.
With SSM as song structure representation, we successfully demonstrate that it is possible to generate drum-track with long-term consistency. This project uses Google's [TensorFlow](https://www.tensorflow.org/ "link") to make the implementation easier.

## The Flow of Generation
![Generation Flow](misc/generation_flow.png "Generation Flow")

- **Data Pre-Processing**: use script
- **Bar Selection**: use script
- **Drum Pattern Generation**: use script



Listening example is available [here](https://sma1033.github.io/drum_generation_with_ssm/ "link"). (It may take a while for reading audio file.)


## Environment
1. Ubuntu 16.04
2. Jupyter Notebook
3. Python 3.5+ 
4. Tensorflow 1.13+
5. Librosa

## To Be Continued