import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("blooms_taxonomy_dataset.csv")
bloom_map = {
    "BT1": "Remember",
    "BT2": "Understand",
    "BT3": "Apply",
    "BT4": "Analyze",
    "BT5": "Evaluate",
    "BT6": "Create"
}
df["Level_Name"] = df["Category"].map(bloom_map)
X = df["Questions"].values
y = df["Category"].values

# Embed questions
model = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = model.encode(X, normalize_embeddings=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, stratify=y, random_state=42)

# Logistic Regression
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[bloom_map[c] for c in clf.classes_]))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Purples",
            xticklabels=[bloom_map[l] for l in clf.classes_],
            yticklabels=[bloom_map[l] for l in clf.classes_])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()
