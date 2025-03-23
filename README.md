# ML-Quarter-2-Project
ML Quarter 2 Project 

Imbalanced datasets can create many issues in machine learning classification algorithms, particularly when the misclassification varies largely across classes. For example, when diagnosing medical cases, false negatives can lead to much more severe consequences when compared to false positives. This would lead to a patient thinking that they don’t have cancer, when they in fact do have cancer. Normally, decision trees are often biased towards the majority class since it splits nodes in a way that reduces overall errors. Since the majority class is more dominant, the algorithm may lean more towards it. 

Our projectgoal is to explore how we can use cost-sensitive learning in decision trees to improve the classification accuracy on imbalanced datasets. We will be using a cost matrix to minimize the misclassification costs. Our primary dataset is the Credit Card Fraud Detection dataset. It contains transactions made by credit cards in September 2013 by European cardholders. The input to our algorithm are the different instances with the transaction features (anonymous to protect privacy) and we used decision trees to output a binary classification of 0 (non-fraudulent) and 1 (fraudulent).




