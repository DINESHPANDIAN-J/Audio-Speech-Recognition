import os
import csv
import torchaudio
import pandas as pd
from jiwer import wer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load the pre-trained model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Function to resample audio to 16kHz
def resample_audio(audio_array, orig_sample_rate=32000, target_sample_rate=16000):
    return torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)(audio_array)

# Function to get transcription using the provided model
def get_transcription(audio_array, sampling_rate):
    target_sample_rate = 16000  # Define the target sampling rate
    audio_array_resampled = resample_audio(audio_array, orig_sample_rate=sampling_rate, target_sample_rate=target_sample_rate)
    
    inputs = processor(
        audio_array_resampled.squeeze(), 
        sampling_rate=target_sample_rate, 
        return_tensors="pt", 
        padding=True
    )

    with torch.no_grad():
        logits = model(inputs.input_values.to(device))
    
    predicted_ids = torch.argmax(logits.logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription

    '''This function get_transcription processes audio data by resampling it to a specific rate,
    then passes it through a pretrained model to predict text information from the audio.
    Finally, it returns the predicted transcription after decoding the model's output.'''

# Function to process all audio files in a folder and store in a CSV file
def process_audio_folder_to_csv(folder_path, output_csv):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]  # Change the extension as needed
    transcriptions = {}
    
    for audio_file in audio_files:
        file_path = os.path.join(folder_path, audio_file)
        
        # Read audio file
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Get transcription for the current audio file
        transcription = get_transcription(waveform, sample_rate)
        
        transcriptions[audio_file] = transcription
    
    # Write transcriptions to CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File Name', 'Transcription'])  # Write header
        
        for audio_file, transcription in transcriptions.items():
            writer.writerow([audio_file, transcription])

# Path to your audio folder in Google Drive
folder_path = "/content/drive/MyDrive/common_voice_test"

# Path to save the output CSV file
output_csv = "/content/drive/MyDrive/transcriptions.csv"

if os.path.exists(folder_path):
    process_audio_folder_to_csv(folder_path, output_csv)
    print(f"Transcriptions saved to {output_csv}")
else:
    print("Folder path does not exist.")

# Load original and model transcriptions from CSV files
original_transcriptions_path = "/path/to/original_transcriptions.csv"
model_transcriptions_path = "/path/to/model_transcriptions.csv"
output_csv_path = "/path/to/output_evaluations.csv"  

original_df = pd.read_csv(original_transcriptions_path)
model_df = pd.read_csv(model_transcriptions_path)

# Merge dataframes on 'File Name' column
merged_df = pd.merge(original_df, model_df, on='File Name', how='inner')

# Calculate WER for each pair of transcriptions and add it to the dataframe
wer_scores = []
for _, row in merged_df.iterrows():
    original_transcription = str(row['Original Transcription_x'])
    model_transcription = str(row['Model Transcription'])
    wer_score = wer(original_transcription, model_transcription)
    wer_scores.append(wer_score)

merged_df['WER'] = wer_scores

# Save the merged dataframe with WER scores to a new CSV file
merged_df.to_csv(output_csv_path, index=False)
