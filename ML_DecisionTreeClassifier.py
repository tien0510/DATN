import pandas as pd
import numpy as np

import category_encoders as ce
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



# đọc tập dữ liệu đầu vào
df = pd.read_csv("data.csv")
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

# convert dữ liệu đưa dữ liệu từ text sang number

encoder = ce.OrdinalEncoder(cols=['Gender','Stream'])
X = encoder.fit_transform(X)

# print(X)
# exit()
# xtrain, xtest, ytrain, ytest = train_test_split(X,y,train_size = 0.8, shuffle = False)


#Xay dung mo hinh DecisionTreeClassifier
# Implementing cross validation với K = 8
k = int(input('Enter k  = '))
kf = KFold(n_splits=k, shuffle=False)  # sử dụng K-Fold để kiểm tra độ chính xác

acc_score = []
best_score = 0
i = 1

#làm tròn tới chữ số thập phân thứ 2
def rnd(i):
    ip = round(i, 2)
    return ip


for train_index, test_index in kf.split(X):
    model = DecisionTreeClassifier()
    # print('test index :,', test_index)
    # print('train index :', train_index)
    X_test, y_test = X.iloc[test_index, :], y[test_index]
    X_train, y_train = X.iloc[train_index, :], y[train_index]

    # Tiến hành huấn luyện mô hình
    model.fit(X_train, y_train)
    pred_values = model.predict(X_test)

    #Dùng các độ đo để kiểm tra mô hình dự đoán
    precision = precision_score(pred_values, y_test, average='macro')
    recall = recall_score(pred_values, y_test, average='macro')
    f_score = f1_score(pred_values, y_test, average='macro')

    accuracy = accuracy_score(pred_values, y_test) * 100

    print(i, '-- Tỷ lệ dự đoán chính xác tập test', rnd(accuracy), '%')
    print("Precision     : ", precision )
    print("Recall        : ", recall )
    print("F1-score      : ", f_score )

    acc_score.append(accuracy)
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        turn = i

        best_precision = precision_score(pred_values, y_test, average='macro')
        best_recall = recall_score(pred_values, y_test, average='macro')
        best_f_score = f1_score(pred_values, y_test, average='macro')

        y_best = y_test
        pred_values_best = pred_values
    i = i + 1

avg_acc_score = sum(acc_score) / k
print('\nMô hình Decision Tree Classifier với k-fold cross-validation :')
print('+++ Tỷ lệ dự đoán chính xác trung bình  : ', rnd(avg_acc_score), '%')
print('+++ Tỷ lệ dự đoán chính xác nhất : ', rnd(best_score), '%', 'vào lần thứ ', turn)
print('\nChất lượng của mô hình dựa trên 3 độ đo :')
print("+++ Precision     : ", best_precision)
print("+++ Recall        : ", best_recall )
print("+++ F1-score      : ", best_f_score )


d = 0
y_best = np.array(y_best)
for i in range(len(pred_values_best)):
        if pred_values_best[i] == y_best[i]:
            d = d + 1
        # print(i,'Dự đoán :', output_DTC[i], ', Thực tế :', y_test[i])


rate_DTC  = round((d/len(pred_values_best))*100,2)
# print('Decision Tree Classifier cho ta tỉ lệ dự đoán  : ')
print('Số dự đoán đúng',d ,'trên tổng',len(pred_values),'.Tỷ lệ chính xác :' ,rate_DTC,'%\n')



import sys
from PyQt5.QtWidgets import QApplication,QMainWindow
from LNqtdesign import Ui_Dialog

class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.uic = Ui_Dialog()
        self.uic.setupUi(self.main_win)

        self.uic.btnresult.clicked.connect(self.showresult)
    def showresult(self):

        age = self.uic.sbage.value()
        gender = self.uic.cbgender.currentIndex()
        gender = gender + 1
        stream = self.uic.cbstream.currentIndex()
        stream = stream + 1
        intership = self.uic.sbintership.value()
        cpga = self.uic.sbcgpa.value()
        hostel = self.uic.cbhostel.currentIndex()
        hob = self.uic.cbhob.currentIndex()

        input = np.array(
            [[age, gender, stream, intership, cpga, hostel, hob]]
        )

# Dùng mô hình có kết quả tốt nhất để đưa vào dự đoán :best model
        output = best_model.predict(input)[0]
        if(output == 'YES') :

             output = 'Congratulations, you got the job'

        else :
            output = "Sorry, you didn't get the job"

        self.uic.Screen.setText(output)

    def show(self):
        self.main_win.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())



