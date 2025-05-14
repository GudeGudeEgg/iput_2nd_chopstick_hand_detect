import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC
import joblib

# X:21個の手のデータ, red_point, blue_point, green_point, yellow_point : 25個のデータ

with open("X.txt", mode="r") as fx:
    X1 = [eval(line.strip()) for line in fx]

with open("Y.txt", mode="r") as fy:
    y = [int(line.strip()) for line in fy]

X1 = np.array(X1)
y = np.array(y)

pre_X = np.zeros((len(X1), 25, 3))
X = np.zeros((len(X1), 18))

for i in range(len(X1)):
    pre_X[i, :, :] = X1[i].reshape(25, 3)


# 0(手首)の座標で正規化
for i in range(len(X1)):
    pre_X[i, :, :] = pre_X[i, :, :] - pre_X[i, 0, :]

x_list = np.zeros((5, 3))

# 指の角度を算出
for i in range(len(X1)):
    x = []
    for j in range(5):
        x_list[0] = pre_X[i, 0, :]
        x_list[1] = pre_X[i, 4 * j + 1, :]
        x_list[2] = pre_X[i, 4 * j + 2, :]
        x_list[3] = pre_X[i, 4 * j + 3, :]
        x_list[4] = pre_X[i, 4 * j + 4, :]
        if j == 0:
            x.append(
                np.dot(x_list[0] - x_list[2], x_list[3] - x_list[2]) / np.linalg.norm(x_list[0] - x_list[2]) /
                np.linalg.norm(x_list[3] - x_list[2]))
        else:
            x.append(
                np.dot(x_list[0] - x_list[1], x_list[2] - x_list[1]) / np.linalg.norm(x_list[0] - x_list[1]) /
                np.linalg.norm(x_list[2] - x_list[1]))
            x.append(
                np.dot(x_list[1] - x_list[2], x_list[3] - x_list[2]) / np.linalg.norm(x_list[1] - x_list[2]) /
                np.linalg.norm(x_list[3] - x_list[2]))
        x.append(
            np.dot(x_list[2] - x_list[3], x_list[4] - x_list[3]) / np.linalg.norm(x_list[4] - x_list[3]) /
            np.linalg.norm(x_list[2] - x_list[3]))
    x_5 = (pre_X[i, 21, :2] + pre_X[i, 22, :2]) / 2
    x_6 = (pre_X[i, 23, :2] + pre_X[i, 24, :2]) / 2
    x.append(
        np.dot(pre_X[i, 4, :2] - x_5, pre_X[i, 8, :2] - x_5) / np.linalg.norm(pre_X[i, 4, :2] - x_5) /
        np.linalg.norm(pre_X[i, 8, :2] - x_5))
    x.append(
        np.dot(pre_X[i, 8, :2] - x_5, pre_X[i, 12, :2] - x_5) / np.linalg.norm(pre_X[i, 8, :2] - x_5) /
        np.linalg.norm(pre_X[i, 12, :2] - x_5))
    x.append(
        np.dot(pre_X[i, 12, :2] - x_6, pre_X[i, 16, :2] - x_6) / np.linalg.norm(pre_X[i, 12, :2] - x_6) /
        np.linalg.norm(pre_X[i, 12, :2] - x_6))

    for j in range(len(x)):
        X[i, j] = x[j]

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# svm_model = SVC(probability=True, C=1.0, random_state=42)
rforest_model = RandomForestClassifier(n_estimators=100, random_state=42)
# gboosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

#ensemble_model = VotingClassifier(estimators=[("rforest", rforest_model), ("gboosting", gboosting_model), ("svm", svm_model)], voting="soft")

# ensemble_model.fit(X_scaled, y_train)
"""
# svmモデルの構築
# svm_model = SVC(kernel="linear", C=1.0, random_state=42)
# svm_model.fit(X_scaled, y_train)
# モデルの評価

rforest_model.fit(X_scaled, y_train)

# y_pred = ensemble_model.predict(X_test_scaled)
# y_pred = svm_model.predict(X_test_scaled)
y_pred = rforest_model.predict(X_test_scaled)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# joblib.dump(ensemble_model, 'ensemble.joblib')
# joblib.dump(svm_model, "svm.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(rforest_model, "rforest.joblib")

