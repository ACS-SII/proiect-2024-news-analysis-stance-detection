import pandas as pd

bert_output = pd.read_csv('Final_Evaluated_Stances.csv')
# bert_output = pd.read_csv("bert_output.csv")
competition_test_stances = pd.read_csv('fnc-1-master/competition_test_stances.csv')

# bert_output['correct'] = bert_output['Stance'].values == competition_test_stances['Stance'].values

merged_df = bert_output.merge(competition_test_stances, on='Body ID', suffixes=('_generated', '_competition'))
merged_df['correct'] = merged_df['Stance_generated'] == merged_df['Stance_competition']
accuracy = merged_df['correct'].mean() * 100
print(f'Procentul de etichete corecte este: {accuracy:.2f}%')

# # bert_output vs competition_test_stances
# accuracy = bert_output['correct'].mean() * 100
# print(f'Procentul de etichete corecte este: {accuracy:.2f}%')