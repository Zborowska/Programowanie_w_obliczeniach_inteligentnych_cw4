import numpy as np
from keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

try:
    # Wczytanie danych z pliku CSV
    file = pd.read_csv('wynik.csv', sep=',')

    # Sprawdzenie czy plik CSV nie jest pusty
    if file.empty:
        print("Plik CSV jest pusty. Upewnij się, że zawiera dane.")
    else:
        # Konwersja danych do tablicy numpy
        data = file.to_numpy()

        # Wydzielenie cech (X) i etykiet (y)
        X = data[1:, :-1].astype('float64')
        y = data[1:, -1]

        # Przygotowanie etykiet do modelu
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        # Podział danych na zbiór treningowy i testowy
        X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoded, test_size=0.3)

        # Tworzenie modelu sieci neuronowej
        model = Sequential()
        model.add(Dense(10, input_dim=72, activation='sigmoid'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

        # Trening modelu
        model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)

        # Ocena dokładności modelu na danych testowych
        loss, accuracy = model.evaluate(X_test, y_test)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

        # Przewidywanie etykiet na danych testowych
        y_pred = model.predict(X_test)
        y_pred_int = np.argmax(y_pred, axis=1)
        y_test_int = np.argmax(y_test, axis=1)

        # Macierz pomyłek
        cm = confusion_matrix(y_test_int, y_pred_int)
        print("Confusion Matrix:")
        print(cm)

        # Wykres krzywej uczenia
        import matplotlib.pyplot as plt

        plt.plot(model.history.history['accuracy'])
        plt.plot(model.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

except FileNotFoundError:
    print("Nie znaleziono pliku CSV. Upewnij się, że ścieżka jest poprawna.")
except IndexError:
    print("Nieprawidłowy zakres danych. Sprawdź, czy używasz właściwego zakresu indeksów.")
except ValueError:
    print("Nieprawidłowy format danych w pliku CSV. Upewnij się, że dane są w oczekiwanym formacie.")
except Exception as e:
    print("Wystąpił błąd:", e)
