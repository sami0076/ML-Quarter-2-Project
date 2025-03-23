#Cost-sensitive learning with Decision Trees

import numpy as np
import pandas as pd

def cost_sensitive_gini(data, target, cost_matrix):
    classes = data[target].unique()
    gini = 0
    for i in classes:
        p_i = data[target].value_counts()[i] / len(data[target])
        for j in classes:
            if i != j:
                gini += p_i * cost_matrix[i][j] * (1-p_i)
    return gini


def cost_sensitive_gini_gain(data, attribute, target, cost_matrix):
    gini_before = cost_sensitive_gini(data, target, cost_matrix)
    gini_after = 0

    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        gini_after += (len(subset) / len(data)) * cost_sensitive_gini(subset, target, cost_matrix)

    return gini_before - gini_after

def findBestAttribute(data, target, cost_matrix):
    best_attribute = None
    best_gain = -float('inf')
    attributes = [col for col in data.columns if col != target]

    for attribute in attributes:
        gain = cost_sensitive_gini_gain(data, attribute, target, cost_matrix)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute

    return best_attribute

def buildTree(data, target, attributes, cost_matrix, depth=0, max_depth=100):
    if depth >= max_depth or len(attributes) == 0 or len(data) == 0:
        return data[target].value_counts().idxmax()

    if len(data[target].unique()) == 1:
        return data[target].iloc[0]

    best_attribute = findBestAttribute(data, target, cost_matrix)
    tree = {best_attribute: {}}

    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        tree[best_attribute][value] = buildTree(subset, target, [attr for attr in attributes if attr != best_attribute], cost_matrix, depth + 1, max_depth)

    return tree

def classifyInstance(tree, instance, data, target):

    if not isinstance(tree, dict):
        return tree
    for attribute, subtree in tree.items():
        value = instance[attribute]
        if value in subtree:
            return classifyInstance(subtree[value], instance, data, target)
        else:
            return 0 #majority class


def buildConfusionMatrix(data, target, tree):
    confusionMatrix = {}
    for value in data[target].unique():
        confusionMatrix[value] = {}
        for value2 in data[target].unique():
            confusionMatrix[value][value2] = 0

    for _, ins in data.iterrows():
        actual = ins[target]
        predicted = classifyInstance(tree, ins, data, target)
        confusionMatrix[actual][predicted] += 1

    return confusionMatrix

def calculateMetrics(confusionMatrix):
    precision = {}
    recall = {}
    f1_score = {}
    for classLabel in confusionMatrix.keys():
        tp = confusionMatrix[classLabel][classLabel]
        fp = sum(confusionMatrix[other][classLabel] for other in confusionMatrix.keys() if other != classLabel)
        fn = sum(confusionMatrix[classLabel][other] for other in confusionMatrix[classLabel].keys() if other != classLabel)
        
        precision[classLabel] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[classLabel] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[classLabel] = (2 * precision[classLabel] * recall[classLabel]) / (precision[classLabel] + recall[classLabel]) if (precision[classLabel] + recall[classLabel]) > 0 else 0
    return precision, recall, f1_score




if __name__ == "__main__":
    target2 = 'Class'
    #Standard data
    trainData = pd.read_csv('cc_finaltraining.csv')
    testData = pd.read_csv('cc_finaltesting.csv')
    #Cost matrix for standard data
    cost_matrix = {
    0: {0: 0, 1:1},
    1: {0: 500, 1: 0},
    }

    '''
    SMOTE data
    trainData = pd.read_csv('cc_final_smote_training.csv')
    testData = pd.read_csv('cc_final_smote_testing.csv')
    #Cost matrix for SMOTE data
    cost_matrix = {
    0: {0: 0, 1:1},
    1: {0: 20, 1: 0},
    }
    '''
    tree = buildTree(trainData, target2, [col for col in trainData.columns if col != target2], cost_matrix)
    print("Tree for credit card dataset (using cost-sensitive Gini index):")
    print(tree)


    trainConfusionMatrix = buildConfusionMatrix(trainData, target2, tree)
    print(f"Training Confusion Matrix: {trainConfusionMatrix}")

    trainPrecision, trainRecall, trainF1 = calculateMetrics(trainConfusionMatrix)
    print(f"Training Precision: {trainPrecision}")
    print(f"Training Recall: {trainRecall}")
    print(f"Training F1 Score: {trainF1}")

    testConfusionMatrix = buildConfusionMatrix(testData, target2, tree)
    print(f"Testing Confusion Matrix: {testConfusionMatrix}")

    testPrecision, testRecall, testF1 = calculateMetrics(testConfusionMatrix)
    print(f"Testing Precision: {testPrecision}")
    print(f"Testing Recall: {testRecall}")
    print(f"Testing F1 Score: {testF1}")