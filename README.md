## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="243" height="192" alt="image" src="https://github.com/user-attachments/assets/23962217-7a17-4803-a252-b593eee76cc0" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

```
<img width="414" height="364" alt="image" src="https://github.com/user-attachments/assets/a45283b5-ce0e-47d7-98f5-d9aa6b5bf954" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="465" height="368" alt="image" src="https://github.com/user-attachments/assets/2ef3e7c5-eb14-49eb-a8a4-131c6167ee5c" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="455" height="365" alt="image" src="https://github.com/user-attachments/assets/2214a6bc-bfb4-403c-a9b8-a1304bf6aab5" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]).toarray())
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="518" height="368" alt="image" src="https://github.com/user-attachments/assets/9a692ee8-d466-4ed5-a946-1714f44dc630" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="716" height="366" alt="image" src="https://github.com/user-attachments/assets/75aaa82e-819e-4dca-9ad0-077c5c24e0f7" />

```
pip install --upgrade category_encoders
```
<img width="1251" height="342" alt="image" src="https://github.com/user-attachments/assets/76e9e1bb-15b9-40be-9ed8-e721d39670b9" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="752" height="362" alt="image" src="https://github.com/user-attachments/assets/4e3d9af6-9fbd-41e8-bdfb-4ee2a5281865" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="623" height="365" alt="image" src="https://github.com/user-attachments/assets/9d292c28-b6ea-43a7-b8a4-83951cc58a5d" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="844" height="429" alt="image" src="https://github.com/user-attachments/assets/aa3b653b-d1c2-4b87-ab37-6f735a0620e2" />

```
df.skew()
```
<img width="336" height="213" alt="image" src="https://github.com/user-attachments/assets/6908e25b-5897-4047-953a-dd02ffd17491" />

```
np.log(df["Highly Positive Skew"])
```
<img width="327" height="461" alt="image" src="https://github.com/user-attachments/assets/fd37fb13-3fda-497b-a935-48a7a784f79a" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="387" height="460" alt="image" src="https://github.com/user-attachments/assets/c27c3337-e683-4c6b-86f9-cffaa9b12628" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="354" height="450" alt="image" src="https://github.com/user-attachments/assets/fda32f57-ddb4-4017-acf6-f364d044c168" />

```
np.square(df["Highly Positive Skew"])
```
<img width="321" height="457" alt="image" src="https://github.com/user-attachments/assets/c7215e60-7ccb-47d7-973e-3c83663cb761" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="849" height="443" alt="image" src="https://github.com/user-attachments/assets/8b0d092c-dfec-4beb-bf99-9ba07a0411f9" />

```
df.skew()
```
<img width="367" height="240" alt="image" src="https://github.com/user-attachments/assets/58027241-dc03-402c-aaa0-19086435295c" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="403" height="272" alt="image" src="https://github.com/user-attachments/assets/7b73334e-c675-4319-bec0-645cd6384d8d" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="845" height="456" alt="image" src="https://github.com/user-attachments/assets/d5813da3-af4f-4efb-b1e2-a7b02cffa882" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="658" height="439" alt="image" src="https://github.com/user-attachments/assets/6a05ee01-e290-49c5-b295-1d1184b11922" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="666" height="453" alt="image" src="https://github.com/user-attachments/assets/bfb85e76-5d6a-4d6e-82be-76d6d600bc0d" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="649" height="446" alt="image" src="https://github.com/user-attachments/assets/8a212e98-6878-43e3-97d4-808a3d236ac5" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="661" height="456" alt="image" src="https://github.com/user-attachments/assets/2814a314-006c-453b-a253-09e64fdad7ee" />

```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
<img width="1233" height="421" alt="image" src="https://github.com/user-attachments/assets/e22739b5-212a-4539-9e39-122c62584fb2" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="666" height="453" alt="image" src="https://github.com/user-attachments/assets/e6e67a7d-f75d-4312-bd41-8fccb9c29f14" />


# RESULT:
       # INCLUDE YOUR RESULT HERE

       
