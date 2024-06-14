import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def get_best_solution(ant_solutions: np.ndarray, X, Y) -> np.array:
    accuracies = np.zeros(ant_solutions.shape[0], dtype=np.float64)
    best_solution = 0
    for i, solution in enumerate(ant_solutions):
        instances_selected = np.nonzero(solution)[0]
        X_train = X[instances_selected, :]
        Y_train = Y[instances_selected]
        classifier_1nn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)
        Y_pred = classifier_1nn.predict(X)
        accuracy = accuracy_score(Y, Y_pred)
        accuracies[i] = accuracy
        if accuracy > accuracies[best_solution]:
            best_solution = i

    # print(f"The winner is ant {best_solution} with accucarcy {accuracies[best_solution]}")
    return ant_solutions[best_solution]

def main():
    print("gerando melhor solução...")
    classe = "stroke"
    base = "datasets"
    original_df = pd.read_csv(base + "/brain-stroke.csv", sep=';')
    dataframe = pd.read_csv(base + "/brain-stroke.csv", sep=';')
    solutions = pd.read_csv("/solutions.csv", sep=';')
    classes = dataframe[classe]
    dataframe = dataframe.drop(columns=[classe])
    indices_selected = np.nonzero(get_best_solution(solutions.to_numpy(), dataframe.to_numpy(), classes.to_numpy()))[0]
    reduced_dataframe = original_df.iloc[indices_selected]
    reduced_dataframe.to_csv(base + '/brain-stroke-c++-2.csv', sep=';', index=False)


'pb'
if __name__ == '__main__':
    main()