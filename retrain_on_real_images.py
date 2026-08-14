import os
import re
import numpy as np
import pickle
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

PROJ = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 64


def extract_features(img_path):
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img)

    hist_feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=8, range=(0, 255))
        hist_feats.extend(hist / hist.sum())

    gray = rgb2gray(arr)
    brightness_mean = gray.mean()
    brightness_std = gray.std()

    gray_u8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray_u8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    dark_ratio = (gray < (brightness_mean - 0.15)).mean()

    return np.array(hist_feats + [brightness_mean, brightness_std, contrast, homogeneity, energy, dark_ratio])


def source_group(fname):
    """Extract the source-photo index (e.g. 'src03_v07.jpg' -> 3) so augmented
    variants of the same original photo are always kept together in one fold."""
    m = re.match(r"src(\d+)_v\d+", fname)
    return int(m.group(1)) if m else fname


def build_dataset(class_dirs):
    X, y, groups = [], [], []
    for label, folder in class_dirs.items():
        for fname in sorted(os.listdir(folder)):
            X.append(extract_features(os.path.join(folder, fname)))
            y.append(label)
            # make groups unique per class+source so healthy-src0 != diseased-src0
            groups.append(f"{label}_{source_group(fname)}")
    return np.array(X), np.array(y), np.array(groups)


def evaluate_group_cv(X, y, groups, task_name):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    logo = LeaveOneGroupOut()
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='rbf', probability=True, random_state=1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    print(f"\n=== {task_name} (Leave-One-Source-Photo-Out CV, {len(set(groups))} groups) ===")
    accs = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, groups=groups, cv=logo, scoring='accuracy')
        accs[name] = scores.mean() * 100
        print(f"{name}: mean accuracy = {scores.mean()*100:.1f}%  (per-fold: {np.round(scores*100,0)})")

    best_name = max(accs, key=accs.get)
    print("\nBest (by group-CV):", best_name)

    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X_scaled, y)

    return final_model, scaler, accs


# ---------------- Hen health ----------------
hen_dirs = {1: f"{PROJ}/dataset/hens/healthy", 0: f"{PROJ}/dataset/hens/diseased"}
X_hen, y_hen, g_hen = build_dataset(hen_dirs)
print("Hen dataset:", X_hen.shape, "class balance:", np.bincount(y_hen), "unique source photos:", len(set(g_hen)))
hen_model, hen_scaler, hen_accs = evaluate_group_cv(X_hen, y_hen, g_hen, "Hen Health Classification (1=healthy, 0=diseased)")

# ---------------- Egg quality ----------------
egg_dirs = {1: f"{PROJ}/dataset/eggs/good_quality", 0: f"{PROJ}/dataset/eggs/poor_quality"}
X_egg, y_egg, g_egg = build_dataset(egg_dirs)
print("\nEgg dataset:", X_egg.shape, "class balance:", np.bincount(y_egg), "unique source photos:", len(set(g_egg)))
egg_model, egg_scaler, egg_accs = evaluate_group_cv(X_egg, y_egg, g_egg, "Egg Quality Classification (1=good, 0=poor)")

# ---------------- Save deployment artifacts (trained on ALL available data) ----------------
with open(f"{PROJ}/hen_health_model.pkl", "wb") as f:
    pickle.dump(hen_model, f)
with open(f"{PROJ}/hen_health_scaler.pkl", "wb") as f:
    pickle.dump(hen_scaler, f)

with open(f"{PROJ}/egg_quality_model.pkl", "wb") as f:
    pickle.dump(egg_model, f)
with open(f"{PROJ}/egg_quality_scaler.pkl", "wb") as f:
    pickle.dump(egg_scaler, f)

with open(f"{PROJ}/model_accuracies.pkl", "wb") as f:
    pickle.dump({"hen": hen_accs, "egg": egg_accs}, f)

print("\nSaved: hen_health_model.pkl, hen_health_scaler.pkl, egg_quality_model.pkl, egg_quality_scaler.pkl, model_accuracies.pkl")
