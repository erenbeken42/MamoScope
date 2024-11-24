import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('imdb_top_1000.csv')


plt.figure(figsize=(15, 8))
sns.countplot(data=df, x='Released_Year', hue='Genre', palette='Set2')
plt.title('Yıllara Göre Tür Dağılımı')
plt.xlabel('Yıl')
plt.ylabel('Film Sayısı')
plt.legend(title='Tür')
plt.xticks(rotation=45)
plt.show()

"""
bu verisetinde yaptığım çoğu grafik çok çeşitle değerler olduğu için verimli olmuyor ama yine de veri görselleştirmesi adına beni
geliştiren bir verisetiydi

"""
