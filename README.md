# Audio Transcription and Evaluation Documentation

## Introduction
This document provides an overview and explanation of the code used for audio transcription and subsequent evaluation using the Wav2Vec2 model and the jiwer library.

## Setup and Dependencies
### Libraries Used:
- `os`: Provides functionalities for interacting with the operating system.
- `csv`: Handles CSV file operations.
- `torchaudio`: Utilized for audio loading and transformation tasks.
- `transformers.Wav2Vec2ForCTC`: Wav2Vec2 model for speech-to-text tasks.
- `transformers.Wav2Vec2Processor`: Processor for Wav2Vec2 model inputs.
- `pandas`: Handles data manipulation and CSV file reading.
- `jiwer`: Used for calculating Word Error Rate (WER).

## Code Structure and Functionality
### Audio Transcription
#### Model Initialization:
The code initializes the Wav2Vec2ForCTC model and Wav2Vec2Processor from the facebook/wav2vec2-base-960h checkpoint.
#### Resampling Function:
- `resample_audio(audio_array, orig_sample_rate=32000, target_sample_rate=16000)`: Resamples the audio array to a target sample rate of 16kHz using torchaudio's Resample.
#### Transcription Function:
- `get_transcription(audio_array, sampling_rate)`: Transcribes audio using the Wav2Vec2 model and processor. It resamples the input audio to the target sample rate and retrieves transcriptions using the Wav2Vec2 model.
#### Processing Audio Folder:
- `process_audio_folder_to_csv(folder_path, output_csv)`: Reads audio files from a specified folder path, transcribes each file, and saves the transcriptions to a CSV file.

### Evaluation Using WER
#### Loading Transcriptions:
The code reads original and model transcriptions from CSV files using pandas.
#### Calculating WER:
It merges the dataframes containing original and model transcriptions based on the 'File Name' column and calculates the Word Error Rate (WER) using the jiwer library.
#### Saving Evaluation Results:
Appends the calculated WER scores as a new column to the merged dataframe and saves it to a new CSV file.

## Usage and Customization
### Audio Transcription:
- Change the `folder_path` and `output_csv` variables to specify the input audio folder and desired output CSV file for transcriptions.
- Ensure the audio files in the folder are in the supported format (e.g., .wav).
### Evaluation Using WER:
- Define the paths for the original transcriptions CSV (`original_transcriptions_path`), model transcriptions CSV (`model_transcriptions_path`), and the output evaluation CSV (`output_csv_path`).

## Conclusion
This code allows for bulk transcription of audio files in a specified folder and the subsequent evaluation of model-generated transcriptions against original transcriptions using the Word Error Rate metric.

