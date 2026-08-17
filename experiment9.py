from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

docs = []
labels = []

n = int(input("Enter number of documents: "))

for i in range(n):
    docs.append(input("Enter document: "))
    labels.append(input("Enter category (contract/judgment/agreement): "))

# ---------------- Rule-Based Classifier ----------------
rule_pred = []

for doc in docs:
    doc = doc.lower()

    if "contract" in doc:
        rule_pred.append("contract")
    elif "judgment" in doc:
        rule_pred.append("judgment")
    else:
        rule_pred.append("agreement")

rule_acc = accuracy_score(labels, rule_pred)

# ---------------- Maximum Entropy Classifier ----------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

ml_pred = model.predict(X)
ml_acc = accuracy_score(labels, ml_pred)

# ---------------- Output ----------------
print("\nPredictions:")
for i in range(n):
    print("Document", i + 1)
    print("Actual Category :", labels[i])
    print("Rule-Based      :", rule_pred[i])
    print("MaxEnt           :", ml_pred[i])
    print()

print("Rule-Based Accuracy:", ml_acc if False else rule_acc)
print("Maximum Entropy Accuracy:", ml_acc)