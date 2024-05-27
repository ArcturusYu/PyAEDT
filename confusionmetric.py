import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

cm_display.plot()
plt.show()


Accuracy = metrics.accuracy_score(actual, predicted)#(True Positive + True Negative) / Total Predictions
print(Accuracy)

Precision = metrics.precision_score(actual, predicted)#True Positive / (True Positive + False Positive)
print(Precision)

Recall = metrics.recall_score(actual, predicted)#True Positive / (True Positive + False Negative)
print(Recall)

F1 = metrics.f1_score(actual, predicted)#2 * (Precision * Recall) / (Precision + Recall)
print(F1)

Specificity = metrics.recall_score(actual, predicted, pos_label = 0)#True Negative / (True Negative + False Positive)