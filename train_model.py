import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import save_model

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

# הכנה של ImageDataGenerator עם חלוקה ל־training ו־validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    '/Users/galshemesh/Downloads/Dataset for Fetus Framework/Dataset for Fetus Framework/Set2-Training-Validation Sets ANN Scoring system',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    '/Users/galshemesh/Downloads/Dataset for Fetus Framework/Dataset for Fetus Framework/Set2-Training-Validation Sets ANN Scoring system',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# בניית המודל
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # משימת סיווג בינארית
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# אימון המודל
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# שמירת המודל
model.save("model_standard_vs_nonstandard.h5")
print("✅ המודל נשמר בהצלחה")
