from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("realtor-data.csv")
data.drop(columns = ["brokered_by","zip_code","prev_sold_date"],inplace = True)
data.isna().sum().sum()
data.dropna(inplace = True)
data.isna().sum()
data.duplicated().sum()
data.drop_duplicates(inplace = True)
data.duplicated().sum()

data["state"].value_counts().sort_values(ascending = True).head(8).plot(kind = "bar")

plt.title("States with Least Number of Houses")
plt.xlabel("States")
plt.ylabel("Number of Houses")

data.groupby(["state","city"])["price"].mean().reset_index()
data.select_dtypes(include = "number")
data.select_dtypes(include = "number").corr()["price"]

X = data[["bed","bath","house_size"]]
Y = data["price"]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

Scalar = StandardScaler()
X_train = Scalar.fit_transform(X_train)

joblib.dump(Scalar,"Scalar.pkl")

X_test = Scalar.transform(X_test)

lr = LinearRegression()
lr.fit(X_train,Y_train)

predictions = lr.predict(X_test)
mean_absolute_error(Y_test,predictions)

joblib.dump(lr,"model.pkl")
