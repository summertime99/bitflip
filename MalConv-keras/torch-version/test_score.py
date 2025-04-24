import pandas as pd

df = pd.read_csv('result.csv', header=None, names=["fn", "label", "score"])

benign = df[df['label'] == 1]
malware = df[df['label'] == 0]


ben_acc = (benign['score'] > 0.5).mean()
mal_acc = (malware['score'] <= 0.5).mean()

print(f"Benign Accuracy: {ben_acc:.4f}")
print(f"Malware Accuracy: {mal_acc:.4f}")

