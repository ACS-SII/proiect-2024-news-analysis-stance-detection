import pandas as pd

training_ids_df = pd.read_csv("./data-all-annotations/trainingdata-ids.txt", sep='\t')
training_data_df = pd.read_csv("./data-all-annotations/trainingdata-all-annotations.txt", sep='\t', encoding='ISO-8859-1')
