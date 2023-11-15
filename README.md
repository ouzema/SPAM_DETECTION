# Mini Project: Spam Detection using Pyspark
Welcome to this repository for the mini project on spam detection using Pyspark. This project aims to explore the effectiveness of various machine learning algorithms, namely Naive Bayes, Random Forest, and Logistic Regression, in identifying spam emails using the Spark framework.
## Data
The email dataset contains 5,572 emails from different users, of which 747 are spam and 4,825 are ham. The dataset is used as a validation set to evaluate the models on a different domain.
## Models
Three different models are built and compared using PySpark:

* Naive Bayes: A probabilistic model that assumes independence between the features and calculates the posterior probability of each class given the input.
* Random Forest: An ensemble model that constructs a number of decision trees and aggregates their predictions using majority voting or averaging.
* Logistic Regression: A linear model that estimates the probability of each class using a logistic function and minimizes the cross-entropy loss.
The models are trained using the TF-IDF features of the emails, which represent the term frequency-inverse document frequency of each word in the vocabulary. The models are evaluated using the accuracy, precision, recall, and F1-score metrics on both the email validation set.
## Results
| Model      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| NaiveBayes | 0.9437    | 0.908  | 0.9173   |

| Model                  | Precision | Recall | F1-Score |
|------------------------|-----------|--------|----------|
| RandomForestClassifier | 0.763     | 0.8735 | 0.8145   |

| Model              | Precision | Recall | F1-Score |
|--------------------|-----------|--------|----------|
| LogisticRegression | 0.9832    | 0.9834 | 0.9832   |

The results show that all three models perform well on the email dataset, with logistic regression having the highest accuracy and F1-score followed by naive bayes and random forest.

## Conclusion
This mini project demonstrates how to use PySpark to build and compare different machine learning models for spam detection. The models are trained and tested on a dataset of emails. The results show that the models have high accuracy and F1-score on the email dataset. Future work could explore other features, models, or techniques to enhance the spam detection performance.


