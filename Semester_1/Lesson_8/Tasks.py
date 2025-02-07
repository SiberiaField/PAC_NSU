import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("wells_info_with_prod.csv", sep=',')

df = df.drop(['API', 'operatorNameIHS', 'BasinName', 'StateName', 'CountyName'], axis=1)
df = pd.get_dummies(df, columns=['formation'])

df = df.drop(['SpudDate', 'PermitDate'], axis=1)
df['CompletionDate'] = pd.to_datetime(df['CompletionDate'])
df['FirstProductionDate'] = pd.to_datetime(df['FirstProductionDate'])

x = df.drop(['Prod1Year', 'ProdAll'], axis=1)
y = pd.DataFrame(df['Prod1Year'], columns=['Prod1Year'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

scaler = MinMaxScaler()
num_collums = ["LatWGS84", "LonWGS84", "BottomHoleLatitude", "BottomHoleLongitude", "LATERAL_LENGTH_BLEND", "PROP_PER_FOOT", "WATER_PER_FOOT"]
x_train[num_collums] = scaler.fit_transform(x_train[num_collums])
x_test[num_collums] = scaler.transform(x_test[num_collums])

y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

print(f"\nx_train:\n{x_train}\n")
print(f"y_train:\n{y_train}\n\n")

print(f"x_test:\n{x_test}\n")
print(f"y_test:\n{y_test}\n")