import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Örnek bir veri kümesi
data = {
    'Days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    'Visitors': [120, 80, 90, 150, 200, 300, 250],
    'Bounce_Rate': [70, 65, 60, 62, 75, 80, 85]
}
df = pd.DataFrame(data)

"""
plt.figure(figsize=(8, 5))
plt.bar(df['Days'], df['Visitors'], color='skyblue')
plt.xlabel('Days')
plt.ylabel('Number of Visitors')
plt.title('Daily Visitors')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(df['Days'], df['Visitors'], marker='o', linestyle='-', color='purple')
plt.xlabel('Days')
plt.ylabel('Number of Visitors')
plt.title('Visitors Trend Over the Week')
plt.show()

plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='Days', y='Visitors', marker='o', color='orange')
plt.title('Visitors Trend Over the Week (Seaborn)')
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Days', y='Visitors', palette='viridis')
plt.title('Daily Visitors (Seaborn)')
plt.show()
"""
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Days', y='Visitors', marker='o', label='Visitors', color='blue')
sns.lineplot(data=df, x='Days', y='Bounce_Rate', marker='o', label='Bounce Rate', color='red')
plt.xlabel('Days')
plt.title('Visitors and Bounce Rate Over the Week')
plt.legend()
plt.show()

# Örnek veriyi çoğaltarak dağılım için kullanabiliriz
visitors_data = {
    'Days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] * 5,
    'Visitors': [120, 80, 90, 150, 200, 300, 250, 130, 85, 95, 155, 210, 290, 240, 
                 125, 82, 92, 140, 205, 310, 255, 110, 75, 89, 130, 220, 310, 260, 
                 135, 90, 100, 160, 215, 295, 245]
}
df_visitors = pd.DataFrame(visitors_data)

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_visitors, x='Days', y='Visitors', palette='pastel')
plt.title('Visitor Distribution Over the Week')
plt.show()





