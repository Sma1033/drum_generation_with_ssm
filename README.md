Drum Generation with Self-Similarity Matrix (SSM)
==================

The goal of this project is using machine learning model to generate long drum-track for a given song.
With SSM as song structure representation, we successfully demonstrate that it is possible to generate drum-track with long-term consistency. This project uses Google's [TensorFlow](https://www.tensorflow.org/ "link") and [Jupyter Notebook](https://github.com/jupyter/notebook "link") to make the implementation easier.

## The Flow of Generation
![Generation Flow](misc/generation_flow.png "Generation Flow")

- **Data Pre-Processing**: Melodic-track MIDI data is synthesized into 44.1 kHz mono-channel audio, then, converted and divided into CQT spectrogram on bar basis. Each bar is a 84 x 96 matrix. Drum tracks are down-sampled to 46 x 16 for each bar.
- **Bar Selection**: To encode song structure, 7-nearest bars are identified for every bar-level spectrogram based on drum SSM to provide relevant information for generating rhythm compatible drum patterns.
- **Training**: Selected 8-bar spectrogram are feed into a VAE-based drum pattern generator as input. Symbolic drum track data is used as ground truth to train the model.
- **Generation**: Similar procedure is performed to generate drum patterns for each single bar except inferred drum SSM is applied for bar selection. 

Listening example is available [here](https://sma1033.github.io/drum_generation_with_ssm/ "link"). (may take a while to open.)


## Environment
1. Intel E5 2660 V4 + Nvidia 1080 Ti (CUDA 9.0 + CUDNN 7.5)
2. Ubuntu 18.04
3. Python 3.6
4. Jupyter Notebook
5. Tensorflow 1.14
6. Librosa
7. FluidSynth


## How to use
1. Download all the files into a specific dir, say, ~/drum_generation
2. Download the SountFonts [here](https://drive.google.com/open?id=1XTrXR27cj02kh1Bxs6YvPTosbQdDN_Am "link"), extract and put it into ~/drum_generation
3. Download pre-trained drum generation model according to the guildline in "drum_generator_model/readme.md"
4. Install all python dependencies 
5. Run the Notebook files from step_1_ to step_5_ to generate drum tracks for input MIDI files in "input_midi" folder
6. Use your own DAW to check the generated drum tracks in the MIDI file in "output_midi" folder