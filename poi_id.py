#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from collections import defaultdict
import matplotlib

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

MISSING_VALUE = "NaN"
FINANCIAL_FEATURE_LIST = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', \
                          'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', \
                          'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
EMAIL_FEATURE_LIST = ['to_messages', 'from_poi_to_this_person', 'from_messages', \
                      'from_this_person_to_poi', 'shared_receipt_with_poi']
POI_FEATURE = 'poi'
NEW_FEATURE_RATIO_OF_MSG_FROM_POI = 'ratio_of_msg_from_poi'
NEW_FEATURE_RATIO_OF_MSG_TO_POI = 'ratio_of_msg_to_poi'

features_list = [POI_FEATURE] +  EMAIL_FEATURE_LIST + FINANCIAL_FEATURE_LIST

# 统计嫌疑人数量
def count_poi(data_dict):
    count = 0
    for person in data_dict:
        if data_dict[person][POI_FEATURE]:
            count += 1
    
    return count

# 统计属性缺失值
def count_missing_value(data_dict):
    dict = defaultdict(int)
    for _, person in data_dict.items():
        for key, value in person.items():
            if value == MISSING_VALUE:
                dict[key] += 1
    
    return dict

#绘制散点图
def draw_scatter_plot(data_set, feature_one, feature_two):
    data = featureFormat(data_set, [feature_one, feature_two])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter(x, y)
    
    matplotlib.pyplot.xlabel(feature_one)
    matplotlib.pyplot.ylabel(feature_two)
    matplotlib.pyplot.show()

# 打印具有异常值的个人    
def print_outliner_name(data_set, feature_name, limit):
    for person, value in data_set.items():
        if (value[feature_name] != MISSING_VALUE and value[feature_name] >= limit):
            print("%s的%s特征值异常：%d" % (person, feature_name, value[feature_name]))

#打印所有人的姓名
def print_person_name(data_set):
    print("嫌疑人姓名：")
    for person in data_set.keys():
        print person

#查找属性值都是NaN的人        
def find_person_with_all_missing_value(data_set):
    result = []
    for person, value in data_set.items():
        append = True
        for attr_name, attr_value in value.items():
            if attr_name == 'poi':
                continue
            if (attr_value != MISSING_VALUE):
                append = False
                break
        if append:
            result.append(person)
    
    return result

# 创建新的特征
def create_ratio_of_message(data_set):
    for person in data_set:
        # 创建收到poi消息比率特征
        msg_from_poi = my_dataset[person]['from_poi_to_this_person']
        to_msg = my_dataset[person]['to_messages']
        if msg_from_poi != MISSING_VALUE and to_msg != MISSING_VALUE:
            data_set[person][NEW_FEATURE_RATIO_OF_MSG_FROM_POI] = msg_from_poi / float(to_msg)
        else:
            my_dataset[person][NEW_FEATURE_RATIO_OF_MSG_FROM_POI] = 0
        
        # 创建发送给poi消息比率特征
        msg_to_poi = data_set[person]['from_this_person_to_poi']
        from_msg = data_set[person]['from_messages']
        if msg_to_poi != MISSING_VALUE and from_msg != MISSING_VALUE:
            data_set[person][NEW_FEATURE_RATIO_OF_MSG_TO_POI] = msg_to_poi / float(from_msg)
        else:
            data_set[person][NEW_FEATURE_RATIO_OF_MSG_TO_POI] = 0

#测试分类器得分
def evaluate_clf(grid_search, features, labels, params, iters = 100):
    acc_scores = []
    pre_scores = []
    recall_scores = []
    
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size = 0.3, random_state = i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc_scores += [accuracy_score(labels_test, predictions)] 
        pre_scores += [precision_score(labels_test, predictions)]
        recall_scores += [recall_score(labels_test, predictions)]
        
    print "accuracy: {}".format(np.mean(acc_scores))
    print "precision: {}".format(np.mean(pre_scores))
    print "recall:    {}".format(np.mean(recall_scores))
 
    if (len(params) != 0):
        print("最优参数：")
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))

#测试贝叶斯、SVM、决策树分类器
def test_clf(data_set, features_list):
    
    data = featureFormat(data_set, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)
    # 添加ratio_of_msg_from_poi特征进行测试
    data = featureFormat(my_dataset, features_list + \
                     [NEW_FEATURE_RATIO_OF_MSG_FROM_POI], \
                     sort_keys = True)
    new_labels, new_features = targetFeatureSplit(data)
    new_features = scaler.fit_transform(new_features)
    
    #朴素贝叶斯
    clf = naive_bayes.GaussianNB()
    params = {}
    grid_search = GridSearchCV(clf, params, cv = 5)
    print("评估朴素贝叶斯算法：")
    evaluate_clf(grid_search, features, labels, params)
    print("=======================添加ratio_of_msg_from_poi特征后===========================")
    evaluate_clf(grid_search, new_features, new_labels, params)
    
    #支持向量机
    clf = SVC(class_weight='balanced')
    params = {
            'C': [1e3, 5e3, 1e4, 5e4, 1e5], 
            'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
    grid_search = GridSearchCV(clf, params, cv = 5)   
    print("评估支持向量机算法：")
    evaluate_clf(grid_search, features, labels, params)
    print("=======================添加ratio_of_msg_from_poi特征后===========================")
    evaluate_clf(grid_search, new_features, new_labels, params)
    
    # 决策树
    clf = DecisionTreeClassifier()
    params = { "criterion": ["gini", "entropy"],
               "min_samples_split": [2, 10, 20],
               "max_depth": [None, 2, 5, 10],
               "min_samples_leaf": [1, 5, 10],
               "max_leaf_nodes": [None, 5, 10, 20], }
    grid_search = GridSearchCV(clf, params, cv = 5)
    print("评估决策树算法：")
    evaluate_clf(grid_search, features, labels, params)
    print("=======================添加ratio_of_msg_from_poi特征后===========================")
    evaluate_clf(grid_search, new_features, new_labels, params)
    
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# 数据点总数   
print("数据点总数： %d" % len(data_dict))
poi_count = count_poi(data_dict)
print("嫌疑人总数： %d" % poi_count)
print("非嫌疑人总数： %d" % (len(data_dict) - poi_count))

# 特征数量
valid_features = data_dict[data_dict.keys()[0]].keys()
print("数据集中可用的特征数为：%d，使用的特征数量为：%d" % (len(valid_features), len(features_list)))

# 统计缺失值
missing_value_map = count_missing_value(data_dict)
print("每个特征的缺失值：")
for feature, count in missing_value_map.items():
    print("{:30s}{:>10d}".format(feature, count))

### Task 2: Remove outliers
# 绘制散点图查看是否有异常值
draw_scatter_plot(data_dict, 'salary', 'bonus')
print_outliner_name(data_dict, 'salary', 2.5e7)
# 移除总数，这可能是输入时的错误引起的
data_dict.pop("TOTAL")
draw_scatter_plot(data_dict, 'salary', 'bonus')

#打印所有人姓名
print_person_name(data_dict)

# 移除非人类的数据
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

#删除所有属性都是NaN的人
for person in find_person_with_all_missing_value(data_dict):
    data_dict[person]

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
#创建两个新特征
create_ratio_of_message(my_dataset)
new_features_list = features_list + [NEW_FEATURE_RATIO_OF_MSG_FROM_POI, NEW_FEATURE_RATIO_OF_MSG_TO_POI]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# 仅保留8个得分最高的特征
k = 8
selector = SelectKBest(f_classif, k = k)
selector.fit_transform(features, labels)
print(("得分最高的%d个特征：") % k)
scores = zip(new_features_list[1:], selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse = True)[:k]
print sorted_scores
final_feature_list = list(zip(*sorted_scores)[0])
print("最终选择的特征：")
print(final_feature_list)
final_feature_list = [POI_FEATURE] + final_feature_list

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#测试贝叶斯、SVM、决策树的性能
#test_clf(my_dataset, final_feature_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

data = featureFormat(my_dataset, final_feature_list, sort_keys = True)
labels, final_features = targetFeatureSplit(data)

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(final_features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# Provided to give you a starting point. Try a variety of classifiers.
clf = naive_bayes.GaussianNB()
#clf = SVC(class_weight='balanced', C = 1000, gamma = 0.0001)

#建立Pipeline
pipeline = Pipeline([('minmaxscaler', MinMaxScaler()), ('gaussiannb', clf)])

dump_classifier_and_data(pipeline, my_dataset, final_feature_list)