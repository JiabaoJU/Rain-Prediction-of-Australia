import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, tree, svm
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")


def data_processing(raw_data):
    data = raw_data.drop('RISK_MM', axis=1)

    plt.figure(figsize=(5, 10))
    data['RainTomorrow'].value_counts().plot(kind='bar')
    plt.title('Target')

    # encode 'RainTomorrow', 'RainToday'
    data['RainTomorrow'] = data['RainTomorrow'].apply(lambda x: 1 if x == 'Yes' else 0)
    data['RainToday'] = data['RainToday'].apply(lambda x: 1 if x == 'Yes' else 0)

    # get the list of feature names for cate value
    def get_cate_data(data):
        list_cate = []
        for col in data.columns:
            if data[col].dtype == 'O':
                list_cate.append(col)
        return list_cate

    data_cate = data[get_cate_data(data)]

    # get the list of feature names for num value
    def get_num_data(data):
        list_num = []
        for col in data.columns:
            if data[col].dtype != 'O':
                list_num.append(col)
        return list_num

    data_num = data[get_num_data(data)]

    # deal with outliers
    numerical = [col for col in data.columns if data[col].dtypes != 'O']
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    fig_rain = data.boxplot(column='Rainfall')
    fig_rain.set_title('')
    fig_rain.set_ylabel('Rainfall')
    plt.subplot(1, 3, 2)
    fig_wind = data.boxplot(column='WindSpeed9am')
    fig_wind.set_title('')
    fig_wind.set_ylabel('WindSpeed9am')
    plt.subplot(1, 3, 3)
    fig_wind = data.boxplot(column='WindSpeed3pm')
    fig_wind.set_title('')
    fig_wind.set_ylabel('WindSpeed3pm')
    plt.savefig("outlier.png")

    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    fig = data.Rainfall.hist(bins=10)
    fig.set_xlabel('Rainfall')
    fig.set_ylabel('RainTomorrow')
    plt.subplot(1, 3, 2)
    fig = data.WindSpeed9am.hist(bins=10)
    fig.set_xlabel('WindSpeed9am')
    fig.set_ylabel('RainTomorrow')
    plt.subplot(1, 3, 3)
    fig = data.WindSpeed3pm.hist(bins=10)
    fig.set_xlabel('WindSpeed3pm')
    fig.set_ylabel('RainTomorrow')
    plt.savefig("distribution_outlier.png")

    IQR = data.Rainfall.quantile(0.75) - data.Rainfall.quantile(0.25)
    Lower_fence_rf = data.Rainfall.quantile(0.25) - (IQR * 3)
    Upper_fence_rf = data.Rainfall.quantile(0.75) + (IQR * 3)

    IQR = data.WindSpeed9am.quantile(0.75) - data.WindSpeed9am.quantile(0.25)
    Lower_fence_ws9 = data.WindSpeed9am.quantile(0.25) - (IQR * 3)
    Upper_fence_ws9 = data.WindSpeed9am.quantile(0.75) + (IQR * 3)

    IQR = data.WindSpeed3pm.quantile(0.75) - data.WindSpeed3pm.quantile(0.25)
    Lower_fence_ws3 = data.WindSpeed3pm.quantile(0.25) - (IQR * 3)
    Upper_fence_ws3 = data.WindSpeed3pm.quantile(0.75) + (IQR * 3)

    def max_value(data, variable, top):
        return np.where(data[variable] > top, top, data[variable])

    data['Rainfall'] = max_value(data, 'Rainfall', Upper_fence_rf)
    data['WindSpeed9am'] = max_value(data, 'WindSpeed9am', Upper_fence_ws9)
    data['WindSpeed3pm'] = max_value(data, 'WindSpeed3pm', Upper_fence_ws3)

    # ================= missing values start =================
    # check missing values
    def check_missing_val(data):
        variable = []
        total_val = []
        data_type = []
        total_missing_val = []
        missing_val_rate = []
        for col in data.columns:
            val = data[col]
            variable.append(col)
            data_type.append(val.dtype)
            total_val.append(val.shape[0])
            total_missing_val.append(val.isnull().sum())
            missing_val_rate.append(round(val.isnull().sum() / val.shape[0], 3) * 100)
        # show result
        missing_data = pd.DataFrame({"Variable": variable, "data_type": data_type, "Total_Val": total_val, \
                                     "Total_Missing_Val": total_missing_val, "Missing_Val_Rate": missing_val_rate}) \
            .sort_values("Missing_Val_Rate", ascending=False)
        return missing_data

    data_info = check_missing_val(data)
    data = data.dropna(subset=get_cate_data(data))
    data = data.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am'])

    # dealing with 'Rainfall','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm'
    def visual_num():
        plt.figure(figsize=(70, 70))
        # rainfall
        plt.subplot(6, 3, 1)
        data['Rainfall'].plot()
        plt.box(False)
        plt.title('rainfall')
        # gust
        plt.subplot(6, 3, 2)
        data['WindGustSpeed'].plot()
        plt.box(False)
        plt.title('WindGustSpeed')
        # WindSpeed9am
        plt.subplot(6, 3, 3)
        data['WindSpeed9am'].plot()
        plt.box(False)
        plt.title('WindSpeed9am')
        # WindSpeed3pm
        plt.subplot(6, 3, 4)
        data['WindSpeed3pm'].plot()
        plt.box(False)
        plt.title('WindSpeed3pm')
        # Humidity9am
        plt.subplot(6, 3, 5)
        data['Humidity9am'].plot()
        plt.box(False)
        plt.title('Humidity9am')
        # Humidity9am
        plt.subplot(6, 3, 6)
        data['Humidity3pm'].plot()
        plt.box(False)
        plt.title('Humidity3pm')

    visual_num()
    plt.savefig('foo.png')

    for col in ['Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm']:
        data[col] = data[col].interpolate()

    # dealing with 'MinTemp', 'MaxTemp'
    data['count_temp'] = data['MaxTemp'].isnull() * 1 + data['MinTemp'].isnull() * 1
    data = data[data['count_temp'] != 2]
    temp_t = data[data['count_temp'] == 0].copy()
    temp_t['diff'] = temp_t['MaxTemp'] - temp_t['MinTemp']
    mean_diff = temp_t['diff'].mean()
    for row in range(data.shape[0]):
        if pd.isna(data.iloc[row]['MaxTemp']) == True:
            data.iloc[row]['MaxTemp'] = data.iloc[row]['MinTemp'] + mean_diff
        if pd.isna(data.iloc[row]['MinTemp']) == True:
            data.iloc[row]['MinTemp'] = data.iloc[row]['MaxTemp'] - mean_diff
    data = data.drop('count_temp', axis=1)

    # dealing with 'Temp3pm'，'Temp9am'
    data['count_daytemp'] = data['Temp3pm'].isnull() * 1 + data['Temp9am'].isnull() * 1
    data = data[data['count_daytemp'] != 2]
    temp_dt = data[data['count_daytemp'] == 0].copy()
    temp_dt['diff'] = temp_dt['Temp9am'] - temp_dt['Temp3pm']
    mean_diff_dt = temp_dt['diff'].mean()
    data = data.drop('count_daytemp', axis=1)

    # dealing with 'Pressure9am', 'Pressure3pm'
    data['count_press'] = data['Pressure9am'].isnull() * 1 + data['Pressure3pm'].isnull() * 1
    data = data[data['count_press'] != 2]
    temp_press = data[data['count_press'] == 0].copy()
    temp_press['diff'] = temp_press['Pressure9am'] - temp_press['Pressure3pm']
    mean_diff_press = temp_press['diff'].mean()
    data = data.drop('count_press', axis=1)

    for row in range(data.shape[0]):
        if pd.isna(data.iloc[row]['Temp9am']) == True:
            data.iloc[row]['Temp9am'] = data.iloc[row]['Temp3pm'] + mean_diff_dt
        if pd.isna(data.iloc[row]['Temp3pm']) == True:
            data.iloc[row]['Temp3pm'] = data.iloc[row]['Temp9am'] - mean_diff_dt
        if pd.isna(data.iloc[row]['Pressure9am']) == True:
            data.iloc[row]['Pressure9am'] = data.iloc[row]['Pressure3pm'] + mean_diff_press
        if pd.isna(data.iloc[row]['Pressure3pm']) == True:
            data.iloc[row]['Pressure3pm'] = data.iloc[row]['Pressure9am'] - mean_diff_press
    # ==================== missing values over ========================

    # since correlation is too low
    data = data.drop(['WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1)
    data = data.drop(['Location'], axis=1)
    data = data.drop(['Date'], axis=1)

    # dealing with 'MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm'
    temp_temp = data[['RainTomorrow', 'MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']].copy()
    temp_temp['temp_range'] = temp_temp['MaxTemp'] - temp_temp['MinTemp']
    temp_temp['mean_temp'] = (temp_temp['Temp9am'] + temp_temp['Temp3pm']) / 2
    temp_temp['std_temp'] = temp_temp['Temp9am'] - temp_temp['Temp3pm']
    temp_temp['std_temp'] = temp_temp['std_temp'].apply(lambda x: abs(x))
    cor = temp_temp.corr()
    data['temp_range'] = data['MaxTemp'] - data['MinTemp']
    data = data.drop(columns=['MinTemp', 'MaxTemp', 'Temp9am'], axis=1)

    # dealing with 'Humidity9am', 'Humidity3pm'
    temp_humi = data[['RainTomorrow', 'Humidity9am', 'Humidity3pm']].copy()
    temp_humi['mean-humi'] = (temp_humi['Humidity9am'] + temp_humi['Humidity3pm']) / 2
    temp_humi['std-humi'] = temp_humi['Humidity9am'] - temp_humi['Humidity3pm']
    temp_humi['std-humi'] = temp_humi['std-humi'].apply(lambda x: abs(x))
    cor = temp_humi.corr()
    data = data.drop('Humidity9am', axis=1)

    data = data.dropna(subset=['Temp3pm', 'Pressure9am', 'temp_range'])

    # dealing with 'Pressure9am', 'Pressure3pm'
    temp_press1 = data[['RainTomorrow', 'Pressure9am', 'Pressure3pm']].copy()
    temp_press1['d-value'] = temp_press1['Pressure9am'] - temp_press1['Pressure3pm']
    temp_press1['d-value'] = temp_press1['d-value'].apply(lambda x: abs(x))
    temp_press1['mean-value'] = (temp_press1['Pressure9am'] + temp_press1['Pressure3pm']) / 2
    cor = temp_press1.corr()
    data = data.drop('Pressure3pm', axis=1)

    return data


# Logistic Regression model
def logReg(X_train, y_train):
    test_c = []
    c_list = [0.001, 1, 100]
    for c in c_list:
        class_logreg = LogisticRegression(C=c, max_iter=10, solver='newton-cg')
        class_logreg.fit(X_train, y_train)
        y_pred = class_logreg.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        test_c.append(score)
    plt.plot(c_list, test_c, 's-', color='r')
    plt.xlabel("C")
    plt.ylabel("Accuracy Score")
    plt.savefig("C.png")

    test_iter = []
    iter_list = [1, 10, 100, 500]
    for iter in iter_list:
        class_logreg = LogisticRegression(C=1, max_iter=iter, solver='newton-cg')
        class_logreg.fit(X_train, y_train)
        y_pred = class_logreg.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        test_iter.append(score)
    plt.plot(iter_list, test_iter, 's-', color='r')
    plt.xlabel("max_iter")
    plt.ylabel("Accuracy Score")
    plt.savefig("max_iter.png")

    test_w = []
    class_logreg = LogisticRegression(C=1, max_iter=100, class_weight=None, solver='newton-cg')
    class_logreg.fit(X_train, y_train)
    y_pred = class_logreg.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    test_w.append(score)
    class_logreg = LogisticRegression(C=1, max_iter=100, class_weight='balanced', solver='newton-cg')
    class_logreg.fit(X_train, y_train)
    y_pred = class_logreg.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    test_w.append(score)
    plt.plot(['None', 'balanced'], test_w, 's-', color='r')
    plt.xlabel("weight")
    plt.ylabel("Accuracy Score")
    plt.savefig("weight.png")

    test_solver = []
    solver_list = ['liblinear', 'lbfgs', 'newton-cg']
    for sol in solver_list:
        class_logreg = LogisticRegression(C=1, max_iter=100, solver=sol)
        class_logreg.fit(X_train, y_train)
        y_pred = class_logreg.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        test_solver.append(score)
    plt.plot(solver_list, test_solver, 's-', color='r')
    plt.xlabel("solver")
    plt.ylabel("Accuracy Score")
    plt.savefig("solver.png")

    # start regression
    class_logreg = LogisticRegression(C=1, max_iter=10, solver='newton-cg')
    class_logreg.fit(X_train, y_train)
    return class_logreg


# decision tree model
def DTC(X_train, y_train):
    test_d = []
    for i in range(12):
        DTC = tree.DecisionTreeClassifier(criterion="gini"
                                          , random_state=30
                                          , max_depth=i + 1
                                          )
        DTC = DTC.fit(X_train, y_train)
        score = DTC.score(X_test, y_test)
        test_d.append(score)
    plt.plot(range(1, 13), test_d, color="red")
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy Score")
    plt.savefig("max_depth.png")

    test_split = []
    for i in range(5, 16):
        DTC = tree.DecisionTreeClassifier(criterion="gini"
                                          , random_state=30
                                          , max_depth=8
                                          , min_samples_split=i
                                          )
        DTC = DTC.fit(X_train, y_train)
        score = DTC.score(X_test, y_test)
        test_split.append(score)
    plt.plot(range(5, 16), test_split, color="red")
    plt.xlabel("min_samples_split")
    plt.ylabel("Accuracy Score")
    plt.savefig("min_samples_split.png")

    class_DTC = tree.DecisionTreeClassifier(criterion="gini", random_state=30, max_depth=8, min_samples_split=7)
    class_DTC = class_DTC.fit(X_train, y_train)
    return class_DTC


# random forest model
def RFC(X_train, y_train):
    scores = []
    for i in range(35, 46):
        RFC = RandomForestClassifier(criterion="gini"
                                     , random_state=30
                                     , max_depth=8
                                     , min_samples_split=7
                                     , n_estimators=i
                                     )
        RFC = RFC.fit(X_train, y_train)
        score = RFC.score(X_test, y_test)
        scores.append(score)
    plt.figure(figsize=[20, 8])
    plt.plot(range(35, 46), scores, color="red", label="n_estimators")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy Score")
    plt.savefig("n_estimators.png")

    class_RFC = RandomForestClassifier(criterion="gini", random_state=30, max_depth=8, min_samples_split=7,
                                       n_estimators=35)
    class_RFC = class_RFC.fit(X_train, y_train)
    return class_RFC


# support vector machine
def SVM(X_train, y_train):
    class_SVM = svm.SVC(kernel="rbf", gamma=0.0359381366380464, cache_size=5000, C=12)
    class_SVM = class_SVM.fit(X_train, y_train)
    return class_SVM


# evaluating the model
def test_model(train_x, test_x, train_y, test_y, model0, model1):
    t0 = time.time()
    pred_y = model0.predict(test_x)
    score_0 = accuracy_score(test_y, pred_y)
    time_taken = time.time() - t0
    print("Accuracy score before normalisation: ", score_0)

    scaler = StandardScaler()
    test_x_normalise = scaler.fit_transform(test_x)
    pred_y_normalise = model1.predict(test_x_normalise)

    score_1 = accuracy_score(test_y, pred_y_normalise)
    print("Accuracy score after normalisation: ", score_1)

    print('Training score: {:.4f}'.format(model0.score(train_x, train_y)))
    print('Test score: {:.4f}'.format(model0.score(test_x, test_y)))

    print("[ classification_report ]")
    print(classification_report(test_y, pred_y))
    # visualize confusion matrix with seaborn heatmap
    confusion = metrics.confusion_matrix(test_y, pred_y)
    cm_matrix = pd.DataFrame(data=confusion, columns=['Actual Positive:1', 'Actual Negative:0'],
                             index=['Predict Positive:1', 'Predict Negative:0'])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='GnBu')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('heatmap after Data Process', fontsize=15)
    plt.show()

    return max(score_0, score_1), time_taken


def model_time_score(model10, model20, model30, model40, model11, model21, model31, model41):
    print("*****the result of Decision Tree:")
    s1, t1 = test_model(X_train, X_test, y_train, y_test, model10, model11)

    print("*****the result of Random Forest:")
    s2, t2 = test_model(X_train, X_test, y_train, y_test, model20, model21)

    print("*****the result of Logistic Regression:")
    s3, t3 = test_model(X_train, X_test, y_train, y_test, model30, model31)

    print("*****the result of Support Vector Machine:")
    s4, t4 = test_model(X_train, X_test, y_train, y_test, model40, model41)

    times = [t1, t2, t3, t4]
    scores = [s1, s2, s3, s4]
    time_score = {'Model': ['DTC', 'RFC', 'LogReg', 'SVM'], 'Accuracy': scores, 'Time taken': times}
    df = pd.DataFrame(time_score)

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_title('Model Comparison: Accuracy and Time taken for execution', fontsize=22)
    color = 'tab:green'
    ax1.set_xlabel('Model', fontsize=20)
    ax1.set_ylabel('Time taken', fontsize=20, color=color)
    ax2 = sns.barplot(x='Model', y='Time taken', data=df, palette='summer')
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Accuracy', fontsize=20, color=color)
    ax2 = sns.lineplot(x='Model', y='Accuracy', data=df, sort=False, color=color)
    ax2.tick_params(axis='y', color=color)
    plt.show()


# Reading data csv file
raw_data = pd.read_csv("weatherAUS.csv")
# heatmap before data_process
plt.figure(figsize=(15, 15))
plt.title('heatmap before Data Process', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = sns.heatmap(raw_data.corr(), square=True, annot=True, fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
plt.show()
plt.savefig("heatmap-before.png")

data = data_processing(raw_data)
# heatmap after data_process
plt.figure(figsize=(15, 15))
plt.title('heatmap after Data Process', fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = sns.heatmap(data.corr(), square=True, annot=True, fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)
plt.show()
plt.savefig("heatmap-after.png")

# splitting data into training and testing dataset
X = data.drop(['RainTomorrow'], axis=1)
y = data['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_normalise = scaler.fit_transform(X_train)
X_test_normalise = scaler.fit_transform(X_test)

# evaluating models
DTC0 = DTC(X_train, y_train)
DTC1 = DTC(X_train_normalise, y_train)

RFC0 = RFC(X_train, y_train)
RFC1 = RFC(X_train_normalise, y_train)

logreg0 = logReg(X_train, y_train)
logreg1 = logReg(X_train_normalise, y_train)

SVM0 = SVM(X_train, y_train)
SVM1 = SVM(X_train_normalise, y_train)

# comparison of score and time
model_time_score(DTC0, RFC0, logreg0, SVM0, DTC1, RFC1, logreg1, SVM1)
