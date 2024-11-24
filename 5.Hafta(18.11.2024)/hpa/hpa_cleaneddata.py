import pandas as pd

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


missing_train = train_data.isnull().sum().sort_values(ascending=False)
missing_test = test_data.isnull().sum().sort_values(ascending=False)

# Yüzde 50den fazla eksiklik içeriyorsa modele katkı yapamcayacağını 
# düşündüğümzden siliyoruz burayı hemde korelasyonda işimizi kolaylaştıracak
threshold = 0.5
train_drop_cols = missing_train[missing_train > len(train_data) * threshold].index
test_drop_cols = missing_test[missing_test > len(test_data) * threshold].index

train_data_cleaned = train_data.drop(columns=train_drop_cols)
test_data_cleaned = test_data.drop(columns=test_drop_cols)

train_data_cleaned.to_csv('train_cleaned.csv', index=False)
test_data_cleaned.to_csv('test_cleaned.csv', index=False)

print("Temizlenmiş veriler 'train_cleaned.csv' ve 'test_cleaned.csv' olarak kaydedildi.")
