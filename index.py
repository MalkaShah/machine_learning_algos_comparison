#type:ignore
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score, davies_bouldin_score
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress all warnings
warnings.filterwarnings("ignore")



#STEP 1: Loading Data
dataset = pd.read_csv('ObesityDataSet.csv')
feature_names = [f'Feature{i+1}' for i in range(22)] 
dataframe = pd.DataFrame(dataset)
dataset['BMI'] = dataset['Weight'] / (dataset['Height'] ** 2)


#STEP 2: Cleaning Data
# For numerical columns, you might want to fill NaN values with the mean or median
numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
dataframe[numerical_columns] = dataframe[numerical_columns].fillna(dataframe[numerical_columns].mean())
# For categorical columns, you might want to fill NaN values with the mode
categorical_columns = dataframe.select_dtypes(include=['object']).columns
dataframe[categorical_columns] = dataframe[categorical_columns].fillna(dataframe[categorical_columns].mode().iloc[0])
#print("the data is: ", dataframe.head())
# Save the cleaned data to a new CSV file
cleaned_file_path = 'cleaned_dataset.csv'
dataframe.to_csv(cleaned_file_path, index=False)



# STEP 3: Extract Categorical and Numerical features from the data Set
# Identify categories of columns
categorical_columns = dataframe.select_dtypes(include=['object']).columns
numerical_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns


#STEP 4 Encoding the categorical Data
# Encode categorical features using One-Hot Encoding
dataframe = pd.get_dummies(dataframe, columns=categorical_columns[:-1], drop_first=True)
# Encode the target variable (assuming the last column is the target)
target_column = categorical_columns[-1]
label_encoder = LabelEncoder()
dataframe[target_column] = label_encoder.fit_transform(dataframe[target_column])
#print("After Encoding the target variable" , dataframe.head())

#STEP 5: Applying principle Component Analysis to Extract 6 features

# Assuming the last column is the target and should be excluded from PCA
X = dataframe.iloc[:, :-1].values  # Features
y = dataframe.iloc[:, -1].values   # Target

# Standardize the data ( it transforms the data so that each feature has a mean of 0 and a standard deviation of 1.)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#feature_names = dataframe.columns

#X_test, X_train, Y_test, Y_train = train_test_split (X_scaled,y,test_size=0.3, random_state=34)

# STEP 6: Apply PCA (to extract top 6 features)
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)
loadings = pca.components_.T  
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2','PC3','PC4', 'PC5','PC6'], index=feature_names)
most_influential_features = {}
for pc in loadings_df.columns:
    most_influential_features[pc] = loadings_df[pc].abs().idxmax()
print("Most influential original features for each principal component:")
for pc, feature in most_influential_features.items():
    print(f"{pc}: {feature}")



#Step 7 dividing dataset
X_test, X_train, Y_test, Y_train = train_test_split(X_pca,y, test_size=0.3,random_state=34)

#Step 8 applying Classifiers Models:
reg_model = LogisticRegression()
reg_model.fit(X_train,Y_train)
reg_pred = reg_model.predict(X_test)
reg_accuracy = accuracy_score(Y_test,reg_pred)
print("The accuracy of Logictic Regssion model is: ", reg_accuracy)

D_tree_model = DecisionTreeClassifier(random_state=34)
D_tree_model.fit(X_train,Y_train)
Dec_tree_pred = D_tree_model.predict(X_test)
Dtree_accuracy = accuracy_score(Y_test,reg_pred)
print("The accuracy of Decision Tree  model is: ", Dtree_accuracy)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, Y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(Y_test,svm_pred)
print("The accuracy of SVM  model is: ", svm_accuracy)

randon_Forest_model = RandomForestClassifier(n_estimators=100, random_state=34)
randon_Forest_model.fit(X_train, Y_train)
Randon_Forest_model_pred = randon_Forest_model.predict(X_test)
RFC_accuracy = accuracy_score(Y_test,Randon_Forest_model_pred)
print("The accuracy of Randon Forest classifier model is: ", RFC_accuracy)

base_estimator = DecisionTreeClassifier(max_depth=1)
AdaBoost_Classifier_model = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=34)
AdaBoost_Classifier_model.fit(X_train, Y_train)
AdaBoost_Classifier_model_pred = randon_Forest_model.predict(X_test)
ABC_accuracy = accuracy_score(Y_test,AdaBoost_Classifier_model_pred)
print("The accuracy of Adaboost classifier model is: ", ABC_accuracy)


# Applying Regressor Models
# Select the relevant columns: Age, Height, BMI, and Weight
new_dataframes = dataset[['Age', 'Height', 'BMI', 'Weight']]

# Define features and target variable
X_new = new_dataframes[['Age', 'Height', 'BMI']]
y_new = new_dataframes['Weight']

xn_train, xn_test, yn_train,yn_test = train_test_split (X_new,y_new,test_size=0.3, random_state=34)

lin_Reg = LinearRegression()
lin_Reg.fit(xn_train,yn_train)
LR_prediction = lin_Reg.predict(xn_test)
MSE = mean_squared_error(yn_test,LR_prediction)
print("the mean sqaured error for Linear Regresion is: ", MSE)
DCR = DecisionTreeRegressor(random_state=34)
DCR.fit(xn_train,yn_train)
DCR_prediction = DCR.predict(xn_test)
MSE2 = mean_squared_error(yn_test,DCR_prediction)
print("the mean sqaured error for Decision Tree Regresion is: ", MSE2)
RFR = RandomForestRegressor(n_estimators=100, random_state=34)
RFR.fit(xn_train,yn_train)
RFR_prediction = RFR.predict(xn_test)
MSE3 = mean_squared_error(yn_test,RFR_prediction)
print("the mean sqaured error for Random Forest Regresion is: ", MSE3)
SVR_R = SVR(kernel='linear')
SVR_R.fit(xn_train,yn_train)
SVR_prediction = SVR_R.predict(xn_test)
MSE4 = mean_squared_error(yn_test,SVR_prediction)
print("the mean sqaured error for Support Vector Regressor is: ", MSE4)

#Step9: Applying Kmeans
# Standardize the features
scalernew = StandardScaler()
dataframe_scaled = scalernew.fit_transform(new_dataframes)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=34)
    kmeans.fit(dataframe_scaled)
    wcss.append(kmeans.inertia_)



# Choose the optimal number of clusters (let's assume 3 based on the elbow plot)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
new_dataframes['Cluster'] = kmeans.fit_predict(dataframe_scaled)



# Evaluate the clustering results
silhouette_avg = silhouette_score(dataframe_scaled, new_dataframes['Cluster'])
davies_bouldin_avg = davies_bouldin_score(dataframe_scaled, new_dataframes['Cluster'])

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Index: {davies_bouldin_avg}")

sns.pairplot(new_dataframes, hue='Cluster', diag_kind='kde', palette='Set1')
plt.suptitle('Pairplot of Clusters Based on Age, Height, Weight, and BMI', y=1.02)
plt.show()

# Visualize the clusters in 2D space (using Age and BMI as an example)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='BMI', hue='Cluster', data=new_dataframes, palette='Set1', s=100, alpha=0.6)
plt.title('Clusters based on Age and BMI')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()









