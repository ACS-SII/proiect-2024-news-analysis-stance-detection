import pandas as pd

bert_output = pd.read_csv('bert_output.csv')
competition_test_stances = pd.read_csv('fnc-1-master/competition_test_stances.csv')

bert_output['correct'] = bert_output['Stance'] == competition_test_stances['Stance']

# Calculeaz? procentul de etichete corecte
accuracy = bert_output['correct'].mean() * 100
print(f'Procentul de etichete corecte este: {accuracy:.2f}%')