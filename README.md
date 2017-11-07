# ML-Practice
ML Practice

Use coef in Logistic Regression, p-value in Linear Regression or feature importance in Random Forest to check the contribution of variables.


Think about the following performance measures that can give more insight into the accuracy of the model than traditional classification accuracy:

Confusion Matrix: A breakdown of predictions into a table showing correct predictions (the diagonal) and the types of incorrect predictions made (what classes incorrect predictions were assigned).

Precision: A measure of a classifiers exactness.

Recall: A measure of a classifiers completeness

F1 Score (or F-score): A weighted average of precision and recall.

Especially,

Kappa (or Cohen’s kappa): Classification accuracy normalized by the imbalance of the classes in the data.

ROC Curves: Like precision and recall, accuracy is divided into sensitivity and specificity and models can be chosen based on the balance thresholds of these values.


用于分类：

Accuracy：准确率

Average_precision：This score corresponds to the area under the precision-recall curve.（Note: this implementation is restricted to the binary classification task or multilabel classification task.）

F1：The F1 score can be interpreted as a weighted average of the precision and recall

F1 = 2 * (precision * recall) / (precision + recall)

f1_micro

F1_macro

F1_weighted

F1_samples

Neg_log_loss：也就是交叉验证熵 -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
This is the loss function used in (multinomial) logistic regression and extensions of it such as neural networks

Precision recall 召回率

roc_auc：Compute Area Under the Curve (AUC) from prediction scores

用于回归：

neg_mean_absolute_error 绝对差

neg_mean_squared_error 方差

R2_score R^2 (coefficient of determination) regression score function.
