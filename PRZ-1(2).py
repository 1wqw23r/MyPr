# Определение функции для обучения модели с разными размерами мини батчей
def train_model_with_batch_size(batch_size):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=100,
                       validation_data=(x_val, y_val),
                       callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
                                 ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)])

    return history

# Обучение модели с разными размерами мини батчей
batch_sizes = [16, 32, 64, 128]
histories = []

for batch_size in batch_sizes:
    print(f'Training with batch size: {batch_size}')
    history = train_model_with_batch_size(batch_size)
    histories.append(history)

# Анализ результатов
import matplotlib.pyplot as plt

for i, batch_size in enumerate(batch_sizes):
    plt.plot(histories[i].history['val_accuracy'], label=f'Batch size {batch_size}')

plt.title('Validation Accuracy for Different Batch Sizes')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()