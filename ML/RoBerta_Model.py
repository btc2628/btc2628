from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
import time

start_time = time.time()


data_file = 'Data/NLP_train_data.xlsx'
df_train = pd.read_excel(data_file)

df_train['Label'] = df_train['grade'] - 1

data_file = 'Data/NLP_validation_data.xlsx'
df_val = pd.read_excel(data_file)
df_val['Label'] = df_val['grade'] - 1

texts = df_train['text'].astype(str).tolist()
labels = df_train['Label'].tolist()

tokenizer = RobertaTokenizer.from_pretrained('RoBerta_Large_Model', local_files_only=True)


encoded_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='tf')
labels = tf.constant(labels)

num_labels = 3

checkpoint_path = 'RoBerta_Large_Model'

model = TFRobertaForSequenceClassification.from_pretrained(checkpoint_path, num_labels=num_labels)

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
metrics = ['accuracy']

def ordinal_loss(y_true, y_pred):
    y_pred_probs = tf.nn.softmax(y_pred)
    y_pred_classes = tf.argmax(y_pred_probs, axis=1, output_type=tf.int32)
    return tf.reduce_mean(tf.square(tf.cast(y_true, tf.int32) - y_pred_classes))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_indicies = list(kf.split(np.arange(len(texts))))

def train_and_evaluate():
    all_scores = []
    for train_idx, val_idx in fold_indicies:
        X_train, X_val = texts[train_idx], texts[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=metrics)
        model.fit(X_train, y_val, verbose=0)


model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=metrics)

train_data = tf.data.Dataset.from_tensor_slices((dict(encoded_inputs), labels)).batch(12)

history = model.fit(train_data, epochs=30, batch_size=12)

his_dict = history.history
print(his_dict)

def predict(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="tf")
    outputs = model(inputs)
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=-1)
    return tf.argmax(probabilities, axis=-1).numpy()[0]

df_val['prediction'] = df_val['text'].apply(predict)

accuracy = (df_val['Label'] == df_val['prediction']).mean()
average_error = (df_val['Label'] - df_val['prediction']).abs().mean()
total_absolute_error = (df_val['Label'] - df_val['prediction']).abs().sum()


duration = time.time() - start_time

print(f"Program ran in {duration} seconds")

save_path = 'Roberta_Trained_Model/'
model.save_pretrained(save_path)


