import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# incarcam datele din setul de date
california_dataset = fetch_california_housing()


def get_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def show_data(dataset):
    # print(dataset.keys())
    # print(dataset.DESCR)
    # print(dataset.feature_names)
    # incarcam datele intr-un dataframe de pandas
    df = pd.DataFrame(california_dataset.data, columns=california_dataset.feature_names)

    print(df.head())
    # print(california['AveRooms'])

    df['MEDV'] = dataset.target
    print(df)


def todo1(input_data, target_data):
    percent_training_data = 69
    X_train, X_predict, y_train, y_predict = train_test_split(input_data, target_data,
                                                              test_size=(100 - percent_training_data) / 100,
                                                              random_state=42)
    # adaugam modelul de regresie
    model = LinearRegression()
    model.fit(X_train, y_train)

    # utilizam modelul pentru a realiza predictii pe setul de testare
    model_output = model.predict(X_predict)

    # calculam eroarea de predictie utilizand RMSE
    pe = get_rmse(predictions=model_output, targets=y_predict)
    print(f"Prediction error (RMSE): {pe}")

    plt.subplot(211)
    # model prediction peste setul de date de testare
    t = range(1, len(model_output) + 1)
    plt.plot(t, y_predict, 'b')
    plt.plot(t, model_output, 'g')
    plt.legend(['target', 'prediction'])
    plt.ylabel("Housing prices")

    plt.title("California Dataset median housing value")

    plt.subplot(212)
    # prediction error peste setul de date de testare
    prediction_error = np.sqrt(np.power(model_output - y_predict, 2))
    plt.plot(prediction_error, 'b')
    plt.legend([f"RMSE: {pe}"])

    plt.xlabel("x (samples)")
    plt.ylabel("Prediction error (RMSE)")
    plt.show()


def todo2(input_data, target_data):
    percent_training_data_range = range(90, 50, -1)
    pe_vect = []
    for percent_training_data in percent_training_data_range:
        X_train, X_predict, y_train, y_predict = train_test_split(input_data, target_data,
                                                                  test_size=(100 - percent_training_data) / 100,
                                                                  random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        model_output = model.predict(X_predict)

        pe = get_rmse(predictions=model_output, targets=y_predict)
        print(f"Predict error (RMSE) for {percent_training_data} percent training data {pe}")
        pe_vect.append(pe)
    plt.plot(percent_training_data_range, pe_vect, 'b')
    plt.legend(['Prediction error'])
    plt.xlabel("Percent training data")
    plt.ylabel("Prediction error (RMSE)")
    plt.title("Calidornia Dataset Housing Value")
    plt.show()


input_data = california_dataset.data
target_data = california_dataset.target

show_data(california_dataset)
todo1(input_data, target_data)
# todo2(input_data, target_data)
