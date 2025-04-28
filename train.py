import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация данных
x_test = x_test.astype("float32") / 255.0

# Загрузка модели (укажите правильный путь к вашей модели)
try:
    model = keras.models.load_model('mnist_model.keras')  # или .h5 для старых версий
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit()

# Проверка архитектуры модели
print("\nАрхитектура модели:")
model.summary()

# Проверка точности на тестовых данных
print("\nОценка на тестовых данных:")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Проверка предсказаний на 10 случайных тестовых примерах
print("\nПроверка предсказаний на 10 случайных примерах:")
indices = np.random.choice(len(x_test), 10)

for i, idx in enumerate(indices):
    # Подготовка изображения
    image = x_test[idx]
    image_input = image.reshape(1, 28, 28)
    
    # Предсказание
    prediction = model.predict(image_input, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    true_digit = y_test[idx]
    
    # Вывод результатов
    print(f"\nПример {i+1}:")
    print(f"Предсказано: {predicted_digit} (уверенность: {confidence:.2%})")
    print(f"Истинное значение: {true_digit}")
    print("Сырые предсказания:", [f"{p:.2f}" for p in prediction[0]])
    
    # Визуализация
    plt.figure(figsize=(3,3))
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {true_digit}, Pred: {predicted_digit}\nConf: {confidence:.2%}")
    plt.axis('off')
    plt.show()

# Проверка распределения предсказаний на всем тестовом наборе
print("\nАнализ распределения предсказаний на тестовом наборе...")
all_predictions = model.predict(x_test, verbose=0)
predicted_digits = np.argmax(all_predictions, axis=1)

# Подсчет частоты каждой цифры
unique, counts = np.unique(predicted_digits, return_counts=True)
digit_counts = dict(zip(unique, counts))

print("\nЧастота предсказаний каждой цифры:")
for digit in range(10):
    count = digit_counts.get(digit, 0)
    print(f"Цифра {digit}: {count} раз ({count/len(x_test):.2%})")

# Проверка весов модели
print("\nПроверка весов модели:")
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'get_weights'):
        weights = layer.get_weights()
        if weights:
            print(f"\nСлой {i} ({layer.name}):")
            print(f"Веса: форма {weights[0].shape}")
            print(f"Смещения: форма {weights[1].shape}")
            print(f"Минимум весов: {np.min(weights[0]):.4f}")
            print(f"Максимум весов: {np.max(weights[0]):.4f}")
            print(f"Среднее весов: {np.mean(weights[0]):.4f}")