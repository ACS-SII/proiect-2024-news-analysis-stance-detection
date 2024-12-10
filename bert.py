import tensorflow as tf
import tensorflow_text as text
import tensorflow_hub as hub
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

preprocess_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"

bert_preprocess_model = hub.KerasLayer(preprocess_url)
bert_model = hub.KerasLayer(encoder_url)

text_test = ['nice movie indeed ', 'I really like programming florin Florin']
text_preprocessed = bert_preprocess_model(text_test)
text_preprocessed.keys()
# print(text_preprocessed['input_word_ids'])

bert_result = bert_model(text_preprocessed)
bert_result.keys()
print(bert_result['pooled_output']) # embedding-ul la nivel de propozitie
print(bert_result['sequence_output']) # ne arata embedding-ul la nivel de cuvant

