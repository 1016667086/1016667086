# 演示精确率,召回率,F1,评估报告,混淆矩阵
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, accuracy_score)

y_true = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
y_pred = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
print("准确率",accuracy_score(y_true, y_pred))
print("精确率",precision_score(y_true, y_pred))
print("召回率",recall_score(y_true, y_pred))
print("F1分数",f1_score(y_true, y_pred))
print("混淆矩阵",confusion_matrix(y_true, y_pred))

print(classification_report(y_true, y_pred))
