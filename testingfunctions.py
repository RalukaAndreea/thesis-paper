from sklearn.model_selection import train_test_split
import pandas as pd

data = {
    'Age': [22, 25, 47, 52, 46],
    'Salary': [25000, 32000, 48000, 52000, 50000],
    'Purchased': [0, 1, 1, 0, 1]
}

print(data, '\n')

df = pd.DataFrame(data)
print(df, '\n')

X = df[['Age', 'Salary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print( "X_train:",'\n', X_train,'\n', "X_test: ", "\n", X_test)
print('\n')
print("Y_train:", '\n' , y_train,'\n', "Y_test: ", '\n', y_test)