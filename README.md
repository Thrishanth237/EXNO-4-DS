# Name: Thrishanth E
# REG NO : 212224230291
# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
FEATURE SCALING
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()

```
![image](https://github.com/user-attachments/assets/c9afa78e-2ebb-4e07-95be-931614122181)

```
df_null_sum=df.isnull().sum()
df_null_sum


df.dropna()


max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/9d0b4e9a-2c9e-4ee1-b5e4-9197f88df000)
![image](https://github.com/user-attachments/assets/183a2770-e8b1-4fa2-8a89-4d7b2f98e03f)
![image](https://github.com/user-attachments/assets/de7bb0a3-62e4-49e2-808f-314defaf4f38)

Standard Scaler

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()

```
![image](https://github.com/user-attachments/assets/c1eb84b3-e083-41fe-b0b6-72f9d2b51aae)

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/d0116f44-16ed-4c9a-9c45-99db3c8e1c37)

MIN_MAX SCALING

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/8f9aec65-fda6-4008-96a5-d417e5069056)

ABSOLUTE MAX
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
```
![image](https://github.com/user-attachments/assets/a20109c2-8ced-4091-85cb-e663cfe3e9e7)

```
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/b476444b-41b4-433d-a6b6-af2171043537)


ROBUST SCALING:

```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()
```
![image](https://github.com/user-attachments/assets/cd776fae-f7c8-4e55-9db1-8ff1574d8f6e)

```
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```

![image](https://github.com/user-attachments/assets/98c3a399-c878-412d-a8b9-d48f93daf302)

FEATURE SELECTION

```

import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/0270f7a7-9ec0-4a47-81fc-66c5fcdc8c02)

```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/456300ab-69b6-4d90-a1f5-bde164038f7a)

```

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/15976f0b-3e2e-4bef-8f1f-37389fd6df9e)


```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/0557f95f-ec8e-4d88-877c-27264e3271ff)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/1c3a5b58-b43d-4aa8-9b12-099d73bd73d4)

FILTER METHOD
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/3905bcfb-f572-4a70-8ce1-96f508b7a81c)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/6cbe3111-1e88-4b79-9116-e2fc03b91d4a)


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif

df.dropna(inplace=True)
X = df.drop(columns=['SalStat'])
y = df['SalStat']

k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/6ae04f18-788a-48e1-a2be-5ed6ed10d19a)

MODEL

```
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/1b11ac15-a15d-4447-a29e-d7786a7fd193)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/b8842891-964f-47b2-bf1c-400c85048e6f)

FISHER SQUARE
```
!pip install skfeature-chappers
```
Collecting skfeature-chappers
  Downloading skfeature_chappers-1.1.0-py3-none-any.whl.metadata (926 bytes)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.11/dist-packages (from skfeature-chappers) (1.6.1)
Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (from skfeature-chappers) (2.2.2)
Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from skfeature-chappers) (2.0.2)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas->skfeature-chappers) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas->skfeature-chappers) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas->skfeature-chappers) (2025.2)
Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn->skfeature-chappers) (1.14.1)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn->skfeature-chappers) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn->skfeature-chappers) (3.6.0)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas->skfeature-chappers) (1.17.0)
Downloading skfeature_chappers-1.1.0-py3-none-any.whl (66 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.3/66.3 kB 3.8 MB/s eta 0:00:00
Installing collected packages: skfeature-chappers
Successfully installed skfeature-chappers-1.1.0





```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]



```

![image](https://github.com/user-attachments/assets/daeef1ca-9872-43c9-9126-3fee8eae2c1d)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/cd6f7b44-f6f6-468e-8349-4d6c338add4a)

ANOVA

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif # Import SelectKBest here

df.dropna(inplace=True)
X = df.drop(columns=['SalStat'])
y = df['SalStat']

k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```

![image](https://github.com/user-attachments/assets/84d93871-e9f3-4c15-82c7-a8a7a1c8865b)

WRAPPER METHOD
```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/93165e2e-c781-4c74-bbd5-73b2338fd13a)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

```
![image](https://github.com/user-attachments/assets/5c353c1b-0c9a-48f3-851a-9884b77b4e92)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/d4cf2756-9972-4ba0-a17c-ee209080ab3c)

```
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```

![image](https://github.com/user-attachments/assets/91cd6e96-3b45-4235-9c02-177ec7be4e1f)

```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```

![image](https://github.com/user-attachments/assets/30ec1f6c-c07f-4725-adcf-be9c28946c59)




# RESULT:
      Thus, Feature selection and Feature scaling has been used on thegiven dataset.
