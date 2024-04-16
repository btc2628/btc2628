from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
import time

start_time = time.time()


data_file = 'Data/train_data.xlsx'
df = pd.read_excel(data_file)

#df['Text'] = df.apply(lambda row: f"Subject: {row['subject_desc']}\nPurpose: {row['purpose_text']}\nBody: {row['body']}", axis=1)

df['Label'] = df['grade'] - 1

df_val = df.sample(n=20, random_state=420)
df_train = df.drop(df_val.index)

texts = df_train['text'].astype(str).tolist()
labels = df_train['Label'].tolist()

tokenizer = RobertaTokenizer.from_pretrained('RoBerta_Large_Model/', local_files_only=True)


encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='tf')
labels = tf.constant(labels)

num_labels = 3

checkpoint_path = 'RoBerta_Large_Model/'

model = TFRobertaForSequenceClassification.from_pretrained(checkpoint_path)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
metrics = ['accuracy']
