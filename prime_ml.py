import math
import numpy as n
from sklearn.ensemble import RandomForestClassifier

# --- Accurate label for training (checking if a number is prime) ---
def is_prime_exact(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True

# --- Characteristics of number ---
def features_for(n: int):
    feats = []
    feats.append(n)
    feats.append(math.log(n+1))
    feats.append(math.sqrt(n))
    for m in [2,3,5,7,11,13,17,19,23]:
        feats.append(n % m)
    feats.append(bin(n).count("1"))
    feats.append(len(bin(n))-2)  # Binary length
    return feats

# --- Building a training database ---
X, y = [], []
for n in range(2, 20000):
    X.append(features_for(n))
    y.append(1 if is_prime_exact(n) else 0)

# --- Model training ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# --- Prediction function ---
def predict_prime(n: int):
    feats = [features_for(n)]
    prob = clf.predict_proba(feats)[0][1]
    pred = clf.predict(feats)[0]
    return {"n": n, "ml_pred": bool(pred), "ml_prob": prob, "exact": is_prime_exact(n)}

# --- Use ---
if __name__ == "__main__":
    n = int(input("Enter a number : "))
    result = predict_prime(n)
    print(f"\nNumber : {result['n']}")
    print(f"Model ML: {'prime number' if result['ml_pred'] else 'composite number'} (possibility ={result['ml_prob']:.3f})")
    print(f"real (Detailed check): {'prime number' if result['exact'] else 'composite number'}")
