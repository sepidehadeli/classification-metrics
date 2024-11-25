import numpy as np

# داده‌ها
x = np.arange(1, 13)
y = np.array(["salem", "salem", "salem", "salem", "mashkook", "mashkook", "mashkook", "mashkook", "bimar", "bimar", "bimar", "bimar"])
y_hat = np.array(["salem", "salem", "salem", "bimar", "mashkook", "mashkook", "mashkook", "mashkook", "bimar", "bimar", "bimar", "mashkook"])

#تبدیل داده ها به 0,1,2
classes = np.array(list(set(y) | set(y_hat)))
class_dict = {c: i for i, c in enumerate(classes)}

# cofusion-matrix
conf_matrix = np.zeros((3, 3), dtype=int)
for i in range(len(y)):
    true = y[i]
    pred = y_hat[i]
    conf_matrix[class_dict[true], class_dict[pred]] += 1


#چاپ ماتریس
print("Confusion Matrix:")
print(conf_matrix)

# محاسبه Precision, Recall, F1-Score و Support
#precision=(True Positive)/(True Positive+False Positive)
precision = np.zeros(len(classes))

#recall=(True Positive)/(True Positive+False Negative)
recall = np.zeros(len(classes))

#f1_score=((precision*recall)/((precision+recall))*2
f1_score = np.zeros(len(classes))

#support=False Negative+True Positive
support = np.zeros(len(classes), dtype=int)

for i, cls in enumerate(classes):
    #True Positive
    tp = conf_matrix[i, i]
    #False Positive
    fp = conf_matrix[:, i].sum() - tp
    #False Negative
    fn = conf_matrix[i, :].sum() - tp
    #support
    support[i] = conf_matrix[i, :].sum()
    precision[i] = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall[i] = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0

# نمایش به صورت جدول
print("\nMetrics:")
print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
print("-" * 50)
for i, cls in enumerate(classes):
    print(f"{cls:<10} {precision[i]:<10.2f} {recall[i]:<10.2f} {f1_score[i]:<10.2f} {support[i]:<10}")
