import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# נתיבים
annotation_path = "/Users/galshemesh/Downloads/Dataset for Fetus Framework/ObjectDetection.xlsx"
image_base_path = "/Users/galshemesh/Downloads/Dataset for Fetus Framework/Dataset for Fetus Framework/Set2-Training-Validation Sets ANN Scoring system/Standard"

# קריאת אנוטציות
annotations = pd.read_excel(annotation_path)
print("עמודות הקובץ:")
print(annotations.columns)

# סינון NT
nt_data = annotations[annotations['structure'] == 'NT']
print(f"נמצאו {len(nt_data)} anotations עבור NT")

# סינון רק קבצים שקיימים באמת
existing_files = set(os.listdir(image_base_path))
nt_data = nt_data[nt_data['fname'].isin(existing_files)]
print(f"מתוך אלה נמצאו {len(nt_data)} תמונות NT שבאמת קיימות בתיקייה")

# מעבר על התמונות והצגת מידע
for idx, row in nt_data.iterrows():
    image_path = os.path.join(image_base_path, row['fname'])
    image = cv2.imread(image_path)

    if image is None:
        continue

    print(f"\n✅ תמונה: {row['fname']}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # חילוץ קואורדינטות
    x_min = int(row['w_min'])
    y_min = int(row['h_min'])
    x_max = int(row['w_max'])
    y_max = int(row['h_max'])

    # חישוב גובה NT
    nt_height = y_max - y_min
    print(f"📏 מדד NT: {nt_height} פיקסלים")

    if nt_height > 30:
        print("⚠️ מדד גבוה - סיכון מוגבר (שקול להפנות לבדיקת מעקב)")
    else:
        print("✅ מדד תקין - ללא אינדיקציה לבעיה")

    # הצגת התמונה
    cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.title(f"NT in {row['fname']}")
    plt.axis('off')
    plt.show()

    # תוכל לשים כאן break אם אתה רוצה לעצור אחרי אחת
    # break
