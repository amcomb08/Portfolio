import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Load the wine dataset and set estimators
data = load_wine()
X, y = data.data, data.target

estimators = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Neighbors': KNeighborsClassifier(),
    'SVM': SVC(kernel='linear', random_state=42)
}

# Find the best classifier between each estimator
def find_best_classifier(estimators, X, y, cv):
    best_score = 0
    best_estimator = ''
    for name, estimator in estimators.items():
        score = cross_val_score(estimator, X, y, cv=cv).mean()
        print(f"{name} ({cv}-fold): {score}")
        if score > best_score:
            best_score = score
            best_estimator = name
    return best_estimator, best_score

# Perform 2-fold cross-validation
print("Performing 2-fold cross-validation")
best_estimator_2, best_score_2 = find_best_classifier(estimators, X, y, cv=2)

# Perform 20-fold cross-validation
print("Performing 20-fold cross-validation")
best_estimator_20, best_score_20 = find_best_classifier(estimators, X, y, cv=20)

# Choose the overall best classifier
best_estimator = best_estimator_2 if best_score_2 > best_score_20 else best_estimator_20
best_score = max(best_score_2, best_score_20)
print(f"\nThe overall best classifier is {best_estimator} with a mean accuracy of {best_score}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = estimators[best_estimator]
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Construct a confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion matrix for {best_estimator}:\n{cm}")
