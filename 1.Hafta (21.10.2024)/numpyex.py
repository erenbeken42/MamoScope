import pandas as pd
df = pd.read_csv('data.csv') #kaggle sitesi üzerinden örnek bir dataset indirdim çeşitli kodlar kullanarak bu veriyi konfigure edeecğim.

print(df)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # bazı satırlardaki tarih fotmatları yanlıs girilmiş düzeltmek için

print(df['Date'])

print(df.isnull().sum())# eksik değer bulma fonksiyonu

#df['Calories'].fillna(df['Calories'].mean(), inplace=True) bu kodu kullanamyıorum çünkü chained assigment dan dolayı sistemin orjinal
#dosya üzerinden mi yoksa kopya dataset üzerinde mi değişiklik yaptığını anlamıyormus.
df['Calories'] = df['Calories'].fillna(df['Calories'].mean())    # null değerlerini ortalama ile dolduruyorum.

high_pulse_df = df[df['Pulse'] > 100]

print(high_pulse_df)


print(df['Calories'].describe())


print(f"Ortalama Kalori: {df['Calories'].mean()}")


print(f"Medyan Kalori: {df['Calories'].median()}")

grouped = df.groupby('Date')['Calories'].sum()

print(grouped)


sorted_df = df.sort_values(by='Calories', ascending=False)

print(sorted_df.head())

df['Calories_per_Minute'] = df['Calories'] / df['Duration']

print(df.head())

df.to_csv('cleaned_data.csv', index=False)



