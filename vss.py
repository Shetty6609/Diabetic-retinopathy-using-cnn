from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

# Load data and preprocess
data_path = 'dataset/'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))

img_size_x = 224
img_size_y = 224
data = []
label = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)
        
    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Convert grayscale to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (img_size_x, img_size_y))
            data.append(resized)
            label.append(label_dict[category])
        except Exception as e:
            print('Exception:', e)

data = np.array(data) / 255.0
label = np.array(label)
from keras.utils import to_categorical
new_label = to_categorical(label)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, new_label, test_size=0.1, random_state=42)

# Load VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size_x, img_size_y, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers for classification
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(categories), activation='softmax')(x)

# Create a new model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    epochs=4,
                    batch_size=32,
                    validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
model.save('vss.h5')
print("Test Accuracy:", accuracy)
