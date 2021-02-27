
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report



def logReg(train_x, test_x, train_y, test_y):
    # start regression
    class_logreg = LogisticRegression(C=0.01, solver='liblinear', random_state=0)
    class_logreg.fit(train_x, train_y)
    pred_y = class_logreg.predict(test_x)
    return class_logreg
    # t0 = time.time()


def train_model(train_x, test_x, train_y, test_y, model):

    pred_y = model.predict(test_x)
    score = accuracy_score(test_y, pred_y)

    print("Accuracy score: ", score)
    # print("Time taken: ", time.time() - t0)
    print('Training score: {:.4f}'.format(model.score(train_x, train_y)))
    print('Test score: {:.4f}'.format(model.score(test_x, test_y)))

    # null accuracy: accuracy that could be achieved by always predicting the most frequent class
    null_accuracy = (36986 / (36986 + 9938))
    print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

    confusion = metrics.confusion_matrix(test_y, pred_y)
    # print('Confusion matrix\n', confusion)
    TP = confusion[0, 0]
    TN = confusion[1, 1]
    FP = confusion[1, 0]
    FN = confusion[0, 1]
    # print('True Positives(TP) = ', TP)
    # print('True Negatives(TN) = ', TN)
    # print('False Positives(FP) = ', FP)
    # print('False Negatives(FN) = ', FN)
    # visualize confusion matrix with seaborn heatmap
    cm_matrix = pd.DataFrame(data=confusion, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='GnBu')
    plt.show()

    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    print('Misclassification rate: {0:0.4f}'.format(classification_error))

    precision = TP / float(TP + FP)
    print('Precision rate: {0:0.4f}'.format(precision))

    recall = TP / float(TP + FN)
    print('Recall rate: {0:0.4f}'.format(recall))

    f1_score = 2 * precision * recall / (precision + recall)
    print('F1 score: {0:0.4f}'.format(f1_score))
    print(classification_report(test_y, pred_y))


data = pd.read_csv("corrected_weather.csv")   # Reading data csv file
# print(data.head(10))

# split data into train and test
X = data.drop(['RainTomorrow'], axis=1)
y = data['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False, random_state=42)

# drop the column of 'Location'
X_train = X_train.drop(columns=['Location'])
X_test = X_test.drop(columns=['Location'])

# logistic regression
print("======= start training Logistic Regression model =======\n")
logreg = logReg(X_train, X_test, y_train, y_test)
train_model(X_train, X_test, y_train, y_test, logreg)


