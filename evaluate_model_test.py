from tensorflow.keras.models import load_model

# טוענים את המודל החדש
model = load_model("model_standard_vs_nonstandard.h5")


import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# נתיב למודל המאומן (אם שמרת אותו) או פשוט השתמש במשתנה `model` אם עדיין בזיכרון
# model = load_model('model.h5') ← אם שמרת אותו לקובץ

IMG_SIZE = 128
test_base_path = '/Users/galshemesh/Downloads/fetus_data/test'

results = []

for label in ['Standard', 'Non-standard']:
    folder = os.path.join(test_base_path, label)
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img / 255.0, axis=0)  # מימד באץ׳

            pred = model.predict(img)[0][0]
            results.append({
                'filename': filename,
                'label': label,
                'prediction': float(pred)
            })

# הדפסת תוצאות
for r in results:
    print(f"{r['filename']} | אמת: {r['label']:14} | ניבוי: {r['prediction']:.3f}")
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# חישוב מדדים
y_true = []
y_pred = []

for r in results:
    true_label = 1 if r['label'] == 'Standard' else 0
    predicted_label = 1 if r['prediction'] >= 0.5 else 0

    y_true.append(true_label)
    y_pred.append(predicted_label)

# דיוק
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ דיוק כולל: {acc*100:.2f}%")

# מטריצת בלבול
cm = confusion_matrix(y_true, y_pred)
print("📉 מטריצת בלבול (Confusion Matrix):")
print(cm)

# גרף מטריצת בלבול
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-standard", "Standard"], yticklabels=["Non-standard", "Standard"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("📊 Confusion Matrix")
plt.tight_layout()
plt.show()

# שמירת התחזיות לקובץ
df_results = pd.DataFrame(results)
df_results.to_csv("predictions_report.csv", index=False)
print("📁 קובץ predictions_report.csv נוצר בהצלחה")
