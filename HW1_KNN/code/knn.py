import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        dist_matrix = []
    
        for i in range(len(X)):
            distances = []
            for j in range(len(self.train_X)):
                dist = sum(abs(X[i] - self.train_X[j]))
                distances.append(dist)
            dist_matrix.append(distances)
    
        return np.array(dist_matrix)


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        dist_matrix = []
        for i in range(len(X)):
            distances = np.sum(np.abs(X[i] - self.train_X), axis=1)
            dist_matrix.append(distances)
        
        return np.array(dist_matrix)


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        abs_diff = np.abs(X[:, np.newaxis] - self.train_X)
        distances = np.sum(abs_diff, axis=2)
        return distances


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1] # столбцы - для train
        n_test = distances.shape[0] # строки - для test
        prediction = np.zeros(n_test) # для хранения предсказанных меток классов для каждого тестового примера

        k = self.k # количество ближайших соседей
        for i in range(n_test):
            closest_neighbors = np.argsort(distances[i])[:k] # argsort - возвращает индексы элементов в порядке их сортировки по возрастанию
            neighbor_labels = self.train_y[closest_neighbors] # создается массивом меток классов ближайших соседей для каждого тестового примера
            prediction[i] = np.argmax(np.bincount(neighbor_labels)) # Прогноз равен наиболее часто встречающейся метке среди соседей

        return prediction
    

    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, int)
        k = self.k # количество ближайших соседей
        for i in range(n_test):
            closest_neighbors = np.argsort(distances[i])[:k]
            neighbor_labels = self.train_y[closest_neighbors]
            prediction[i] = np.argmax(np.bincount(neighbor_labels))

        return prediction
