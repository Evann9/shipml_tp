# RandomForest 모델
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "ais_data_with_trig.csv"
TARGET = "shiptype"


def print_top_confusion_pairs(cm_df, top_n=10):
    errors = []
    for actual in cm_df.index:
        for predicted in cm_df.columns:
            if actual != predicted and cm_df.loc[actual, predicted] > 0:
                errors.append((actual, predicted, int(cm_df.loc[actual, predicted])))

    errors.sort(key=lambda x: x[2], reverse=True)

    print(f"\ntop {top_n} confusion pairs:")
    if not errors:
        print("no misclassifications")
        return

    for actual, predicted, count in errors[:top_n]:
        print(f"{actual} -> {predicted}: {count}")


df = pd.read_csv(DATA_PATH)
print("data shape:", df.shape)

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("categorical columns:", categorical_cols)
print("numeric columns:", numeric_cols)
print("target classes:", y.nunique())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("train shape:", X_train.shape, y_train.shape)
print("test shape:", X_test.shape, y_test.shape)

tree_preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical_cols,
        ),
    ]
)

linear_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            categorical_cols,
        ),
    ]
)

model_configs = {
    "logistic_regression": Pipeline(
        steps=[
            ("preprocessor", linear_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    ),
    "random_forest": Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    ),
    "extra_trees": Pipeline(
        steps=[
            ("preprocessor", tree_preprocessor),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    ),
}

results = []
best_name = None
best_model = None
best_test_acc = -1.0

for model_name, pipeline in model_configs.items():
    print(f"\n===== {model_name} =====")
    model = clone(pipeline)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    report = classification_report(y_test, test_pred, output_dict=True)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("train accuracy:", train_acc)
    print("test accuracy:", test_acc)
    print("macro f1:", report["macro avg"]["f1-score"])
    print("weighted f1:", report["weighted avg"]["f1-score"])

    results.append(
        {
            "model": model_name,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
    )

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_name = model_name
        best_model = model

results_df = pd.DataFrame(results).sort_values(by="test_accuracy", ascending=False)

print("\nmodel comparison:")
print(results_df.to_string(index=False))
print(f"\nbest model: {best_name}")

best_pred = best_model.predict(X_test)

print("\nclassification report:")
print(classification_report(y_test, best_pred))

labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print_top_confusion_pairs(cm_df)
