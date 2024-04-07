# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:30:35 2024

@author: saga
"""

#***********************STEP-00: SET VARIABLES *******************************


Input_Video = "https://www.youtube.com/watch?v=GEPhLqwKo6g"
   # APPLE_youtube_url   = "https://www.youtube.com/watch?v=GEPhLqwKo6g"
   # KENNEDY_youtube_url = "https://www.youtube.com/watch?v=0fkKnfk4k40"
    
audio_folder = 'D:/PYTHON/AUDIOS'

#***********************STEP-01: DOWNLOAD VIDEO********************************
from pytube import YouTube
import pandas as pd

#DOWNLOAD
yt = YouTube(Input_Video)
video_name = yt.title
stream = yt.streams.filter(file_extension='mp4').get_highest_resolution()

# RENAME & STORE IN LOCAL PATH
audio_file_name = ''.join(c if c.isalnum() else '_' for c in video_name) + ".mp4"
stream.download(output_path=audio_folder, filename=audio_file_name)

# STORE IN DATAFRAME
df_video = pd.DataFrame({'video_name': [audio_file_name]})

print("Video downloaded OK")

#************************STEP-02: TRANSCRIPTION*******************************

import os
#import pandas as pd
from transformers import pipeline
from datetime import datetime

# TRANSFORMERS MODEL: whisper-small
def transcribe_audio(audio_file_path):
    transcriber = pipeline(task="automatic-speech-recognition", model="openai/whisper-small")
    transcription_results = transcriber(audio_file_path)
    transcription_text = transcription_results.get('text', "No transcription results found.")
    
#SAVE THE DATE WE MADE THE TRANSCRIPTION
    current_date = datetime.now().strftime("%Y-%m-%d")
    
# CREATE DATAFRAME TO STORE RESULTS
    df = pd.DataFrame({'Current_Date': [current_date],
                       'Video_Name': [audio_file_name], 
                       'Transcription': [transcription_text]})
    
# PRINT TRANSCRIPTION
    print("Transcription Results:", transcription_text) 
    return df

#SEARCH FOR THE AUDIO IN THE FILE PATH AND EXECUTE TRANSCRIPTION
audio_file_path = os.path.join(audio_folder, audio_file_name)
df = transcribe_audio(audio_file_path)
print(df)

#***************STEP-03: TRANSLATION MODEL-ROA LIMIT OF 512 *******************

from transformers import MarianMTModel, MarianTokenizer

#SET THE MODEL PARAMETERS
model_name = "Helsinki-NLP/opus-mt-en-roa"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
target_language_code = ">>spa<<"

#TRANSLATE IN BLOCKS OF 512 TOKENS
def translate_text(text):
    segment_length = 512
    segments = [text[i:i+segment_length] for i in range(0, len(text), segment_length)]
    translated_text = ""
    
    for segment in segments:
        input_text = f"{target_language_code} {segment}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        translated_ids = model.generate(**inputs)
        translated_text += tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text.strip()

#ADD TO DATAFRAME
df['Translated_Transcription'] = df['Transcription'].apply(translate_text)


#********PASS TRANSLATED OUTPUT TO ENGLISH ***********************************

target_translation = 'Spanish to English'

# CREATES DATAFRAME WITH LANGUAGE OPTIONS
language_models_df = pd.DataFrame({
    'Language': ['Spanish', 'Mandarin', 'French', 'Hindi', 'Spanish to English'],
    'Model_Name': ["Helsinki-NLP/opus-mt-en-es", "Helsinki-NLP/opus-mt-en-zh", 
                   "Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-en-hi",
                   "Helsinki-NLP/opus-mt-es-en"]})

#SET THE MODEL PARAMETERS
model_name = language_models_df.loc[language_models_df['Language'] == target_translation, 'Model_Name'].values[0]
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

#TRANSLATE IN BLOCKS OF 512 TOKENS
def translate_text(text):
    segment_length = 512
    segments = [text[i:i+segment_length] for i in range(0, len(text), segment_length)]
    translated_text = ""   
    for segment in segments:
        inputs = tokenizer(segment, return_tensors="pt", padding=True)
        translated_ids = model.generate(**inputs)
        translated_text += tokenizer.decode(translated_ids[0], skip_special_tokens=True) + " "
    return translated_text.strip()

#ADD TO DATAFRAME
df['Translated_text_to_eng'] = df['Translated_Transcription'].apply(translate_text)

#***********************STEP-04:TEXT COMPARISSON*******************************

import spacy
import pandas as pd

#spaCy MODEL
nlp = spacy.load("en_core_web_sm")

#EXTRACT TEXTS
text1 = df['Transcription'].iloc[0]  
text2 = df['Translated_text_to_eng'].iloc[0]

# COMPARATIVE ANALYSIS
doc1 = nlp(text1)
doc2 = nlp(text2)
similarity_score = doc1.similarity(doc2)

#SAVE THE RESULTS & PRINT SCORE
translation_metrics = pd.DataFrame({
    'Original_Text': [text1],
    'Translated_text_to_eng': [text2],
    'Similarity_Score': [similarity_score]})
print("Score:", similarity_score)

#***********************STEP 05: EXPORT OUTPUTS******************************

#EXPORT DATAFRAME TO CSV FILE
csv_file_path = os.path.join(audio_folder, 'transcription_results.csv')
df.to_csv(csv_file_path, index=False)
print("DataFrame exported to CSV file")

#EXPORT OUTPUT TO TEXT FILE
txt_file_path = os.path.join(audio_folder, 'translated_transcription.txt')
with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
    for index, row in df.iterrows():
        txt_file.write(f"Title: {row['Video_Name']}\n")
        txt_file.write(row['Translated_Transcription'] + '\n\n')
print("Text exported to TXT file")