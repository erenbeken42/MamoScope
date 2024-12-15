import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Base modelin ağırlıklarını dondurduk çünkü önceden öğrendiği özellikleri tutarak sadece üstüne katman eklemek istiyoruz 
base_model.trainable = False

#base model olarak ResNet50 yi alarak üzerine katmanlar eklemek için modeli kurguladık

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(1, activation='sigmoid') 
])

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory('path_to_train_data', target_size=(224, 224), batch_size=32, class_mode='binary')

model.fit(train_generator, epochs=10, steps_per_epoch=100)

# Fine-tuning: Sonraki katmanları ince ayar yap
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Son 10 katmanı ince ayar yapacağız
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
#learning ratein ne kadar düşük olduğuna bak ince ayar yaptığınmız için daha dikkatli değişiklikler olmasını sağlıyoruz
model.fit(train_generator, epochs=10, steps_per_epoch=100)
