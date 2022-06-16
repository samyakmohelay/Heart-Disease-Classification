# Heart-Disease-Prediction
## Predicting heart disease using machine learning.

This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting weather or not someone has heart disease based on their medical attributes.

### What is classification?
Classification involves deciding whether a sample is part of one class or another (single-class classification). If there are multiple class options, it's referred to as multi-class classification.

### What we'll end up with
Since we already have a dataset, we'll approach the problem with the following machine learning modelling framework.

IMAGE HERE

More specifically, we'll utilize the following steps.

- **Exploratory data analysis (EDA)** - the process of going through a dataset and finding out more about it.
- **Model training** - create model(s) to learn to predict a target variable based on other variables.
- **Model evaluation** - evaluating a models predictions using problem-specific evaluation metrics.
- **Model comparison** - comparing several different models to find the best one.
- **Model fine-tuning** - once we've found a good model, how can we improve it?
- **Feature importance** - since we're predicting the presence of heart disease, are there some things which are more important for prediction?
- **Cross-validation** - if we do build a good model, can we be sure it will work on unseen data?
- **Reporting what we've found** - if we had to present our work, what would we show someone?

To work through these topics, we'll use pandas, Matplotlib and NumPy for data anaylsis, as well as, Scikit-Learn for machine learning and modelling tasks.

We're going to take the following approach:

1. Problem definition
2. Data
3. Evaluation
4. Features
5. Modelling
6. Experimentation

### 1. Problem Definition
In our case, the problem we will be exploring is **binary classification** (a sample can only be one of two things).

This is because we're going to be using a number of different features (pieces of information) about a person to predict whether they have heart disease or not.

In a statement,
> Given clinical parameters about a patient, can we predict whether or not they have heart disease?

### 2. Data
The original data came from the Cleavland data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).

There is also a version of it available on [Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci).

The original database contains 76 attributes, but here 14 of those attributes will be used. Attributes (also called features) are the variables what we'll use to predict our target variable.

### 3. Evaluation
> If we can reach 90% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.

### 4. Features
This is where we'll get different information about each of the features in our data. We can do this via doing our own research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).

#### Create data dictionary

The following are the features we'll use to predict our target variable (heart disease or no heart disease).

1. age - age in years
2. sex - (1 = male; 0 = female)
3. cp - chest pain type
   - 0: Typical angina: chest pain related decrease blood supply to the heart
   - 1: Atypical angina: chest pain not related to heart
   - 2: Non-anginal pain: typically esophageal spasms (non heart related)
   - 3: Asymptomatic: chest pain not showing signs of disease
4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)
   - anything above 130-140 is typically cause for concern
5. chol - serum cholestoral in mg/dl
   - serum = LDL + HDL + .2 * triglycerides
   - above 200 is cause for concern
6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
   - '>126' mg/dL signals diabetes
7. restecg - resting electrocardiographic results
   - 0: Nothing to note
   - 1: ST-T Wave abnormality
     - can range from mild symptoms to severe problems
     - signals non-normal heart beat
   - 2: Possible or definite left ventricular hypertrophy
     - Enlarged heart's main pumping chamber
8. thalach - maximum heart rate achieved
9. exang - exercise induced angina (1 = yes; 0 = no)
10. oldpeak - ST depression induced by exercise relative to rest
    - looks at stress of heart during excercise
    - unhealthy heart will stress more
11. slope - the slope of the peak exercise ST segment
    - 0: Upsloping: better heart rate with excercise (uncommon)
    - 1: Flatsloping: minimal change (typical healthy heart)
    - 2: Downslopins: signs of unhealthy heart
12. ca - number of major vessels (0-3) colored by flourosopy
    - colored vessel means the doctor can see the blood passing through
    - the more blood movement the better (no clots)
13. thal - thalium stress result
    - 1,3: normal
    - 6: fixed defect: used to be defect but ok now
    - 7: reversable defect: no proper blood movement when excercising
14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)

### 5. Modelling 
After Exploring the data, we'll try to use machine learning to predict our target variable based on the 13 independent variables.

We're trying to predict our target variable using all of the other variables. To do this, we'll split the target variable from the rest.

### **Train & Test Split**
This is where we'll split our data into a training set and a test set.

We use our training set to train our model and our test set to test it.
The test set must remain separate from our training set.

### **Model choices**
Now we've got our data prepared, we can start to fit models. We'll be using the following and comparing their results.

- Logistic Regression - LogisticRegression()
- K-Nearest Neighbors - KNeighboursClassifier()
- RandomForest - RandomForestClassifier()

Now we've got a baseline model... and we know a model's first predictions aren't always what we should based our next steps off.

So we'll try to increase the accuracy of our model further by applying the following methods below.

- **Hyperparameter tuning** - Each model we use has a series of dials we can turn to dictate how they perform. Changing these values may increase or decrease model performance.
- **Feature importance** - If there are a large amount of features we're using to make predictions, do some have more importance than others? For example, for predicting heart disease, which is more important, sex or age?
- **Confusion matrix** - Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagnol line).
- **Cross-validation** - Splits your dataset into multiple parts and train and tests your model on each part and evaluates performance as an average.
- **Precision** - Proportion of true positives over total number of samples. Higher precision leads to less false positives.
- **Recall** - Proportion of true positives over total number of true positives and false negatives. Higher recall leads to less false negatives.
- **F1 score** - Combines precision and recall into one metric. 1 is best, 0 is worst.
- **Classification report** - Sklearn has a built-in function called classification_report() which returns some of the main classification metrics such as precision, recall and f1-score.
- **ROC Curve** - Receiver Operating Characterisitc is a plot of true positive rate versus false positive rate.
- **Area Under Curve (AUC)** - The area underneath the ROC curve. A perfect model achieves a score of 1.0.

### 6. Experimentation
If we haven't hit our evaluation metric yet... ask ourselves...

Could we collect more data?
Could we try a better model? Like CatBoost or XGBoost?
Could we improve the current models? (beyond what we've done so far)

And do some experimentations regarding the above questions

### Preparing the tools
We will be using the following libraries,
- **Pandas** for data analysis.
- **NumPy** for numerical operations.
- **Matplotlib/seaborn** for plotting or data visualization.
- **Scikit-Learn** for machine learning modelling and evaluation.
