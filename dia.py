import pandas as pd
import warnings
import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import  LogisticRegression
data = pd.read_csv('diabetes.csv')
pd.set_option('display.max_columns',None)
warnings.filterwarnings("ignore")
#print(data.head())
y=data.pop('Outcome')
print(data.isnull().sum())
#print(data.head())
x_train,x_test,y_train,y_test = train_test_split(data,y,test_size=0.2,stratify= y,random_state=20)
sm=SMOTE()
x_train , y_train = sm.fit_resample(x_train,y_train)
a = y_train[y_train ==0]
print(len(a))
model = LogisticRegression(max_iter = 2000)
scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
model =model.fit(x_train,y_train)
y_train = pd.DataFrame(y_train)
print(y_train.head(10))

fearture_names=data.columns.tolist()
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(fearture_names,"features.pkl")
