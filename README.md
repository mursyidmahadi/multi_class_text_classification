# Multi-class Text Classification
This repository contains all codes where a case study were conducted to classify text articles into five categories : Sport, Tech, Business, Entertainment, Politics. A Natural Language Processing (NLP) are used to solve this problem.

## Task Implementation

## 1. Exploratory Data Analysis
- Check the first five dataset by using print(df.head())
- Check for the text data print(df['text'][1])

## 2. Data Cleaning
- Remove everything in bracket
- temp = re.sub('\(.*?\)', ' ', data)
- Replace all non-characters to spaces
- temp = re.sub('[^a-zA-Z]', ' ', temp)
- Remove singular characters such as 'worldcom s problems' to 'worldcom problems'.
- temp = re.sub('\s[a-z]\\b', '', temp)
- Change all characters to lowercase
- text[i] = temp.lower()

## 3. Data Preprocessing
4 steps were taken for data preprocessing
- Text tokenization
- Text padding & truncation
- OneHotEncoder for target variable
- Train-test split with test size of 0.2

## 4. Model Development
A model architecture is used for this problem.
- Dropout rate is set to 0.4
- Vocabulary size is set to 100
- Softmax Activation Function is used/act as Activation Function
- Categorical Cross-Entropy Function is set to loss function
- No early stopping implemented.

## 5. Results
- The performance of the model and the reports are observed
- Accuracy and F1 Score is recorded and compare for 5 samples
- The confusion matrix is applied to compare the effectiveness of model in between 5 fields.
