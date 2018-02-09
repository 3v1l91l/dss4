import pandas as pd
import matplotlib.pyplot as plt

finder_decisions= pd.read_csv('Finder_decisions_.csv', sep=';')
sender_counts = finder_decisions['Sender_id'].value_counts()
print(sum((sender_counts == 1)))

