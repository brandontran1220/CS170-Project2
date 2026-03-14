import time
import numpy as np
from numba import njit

def load_dataset(filename):
    data = np.loadtxt(filename)
    class_name = data[:, 0]
    features = data[:, 1:]
    return class_name, features

@njit
def euclidean_distance(first, second, feature_set):
    distance = 0.0
    for i in feature_set: ## iterate through the specified features
        distance += (first[i - 1] - second[i - 1]) ** 2 ## calculate the squared difference for each feature and sum them up
    return distance ** 0.5 ## return the Euclidean distance

@njit
def leave_one_out(features, class_name, feature_set):
    number_correctly_classified = 0

    for i in range(len(features)): ## iterate through each instance in the dataset
        object_to_classify = features[i]
        label_object_to_classify = class_name[i]
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = None

        for k in range(len(features)): ## iterate through the dataset again to find the nearest neighbor
            if i == k:
                continue
            
            distance = euclidean_distance(object_to_classify, features[k], feature_set) ## calculate the distance between the two features

            if distance < nearest_neighbor_distance: ## if this instance is closer than the previous nearest neighbor, update the nearest neighbor and distance
                nearest_neighbor_distance = distance
                nearest_neighbor_label = class_name[k]
        if label_object_to_classify == nearest_neighbor_label: ## increment number of correct classifications if the nearest neighbor has the same class name as the current instance
            number_correctly_classified += 1

    return number_correctly_classified / len(features) ## return the accuracy as the percentage of correct classifications 

def forward_selection(features, class_name, cnt_features):
    current_features = [] ## start with an empty set of features for forward selection
    best_features = []
    best_accuracy = 0.0

    print("Beginning search.\n")

    for _ in range(cnt_features): ## iterate through the number of features to add
        feature_to_add = None
        best_temp_accuracy = 0.0

        for feature in range(1, cnt_features + 1): ## iterate through all features to find the best one to add
            if feature in current_features:
                continue
            
            temp_features = current_features + [feature] ## create a temporary set of features by adding the current feature
            curr_accuracy = leave_one_out(features, class_name, temp_features) ## calculate the accuracy with the temporary set of features

            print(f"Using feature(s) {temp_features} accuracy is {curr_accuracy * 100:.1f}%")

            if curr_accuracy > best_temp_accuracy: ## update the best temp accuracy and feature
                best_temp_accuracy = curr_accuracy
                feature_to_add = feature
        
        current_features.append(feature_to_add) ## add the best feature to the current set of features
        print(f"Feature set {current_features} was best, accuracy is {best_temp_accuracy * 100:.1f}%")

        if best_temp_accuracy > best_accuracy: ## if the current set has better accuracy than the best so far, update
            best_features = current_features.copy()
            best_accuracy = best_temp_accuracy
        else:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n")

    return best_features, best_accuracy

def backward_elimination(features, class_name, cnt_features):
    current_features = list(range(1, cnt_features + 1)) ## start with all features for backward elimination
    best_features = current_features.copy()
    best_accuracy = leave_one_out(features, class_name, current_features) ## calculate the accuracy with all features

    print("Beginning search.\n")

    for i in range(cnt_features - 1): ## iterate through the number of features to remove
        target = None
        best_temp_accuracy = 0.0

        for feature in current_features: ## iterate through the current set of features to find the best one to remove
            temp_features = [f for f in current_features if f != feature] ## remove the current feature from the set
            curr_accuracy = leave_one_out(features, class_name, temp_features) ## calculate the accuracy with the current set of features without the removed feature

            print(f"Using feature(s) {temp_features} accuracy is {curr_accuracy * 100:.1f}%")

            if curr_accuracy > best_temp_accuracy: ## update the best temp accuracy and feature to remove
                best_temp_accuracy = curr_accuracy
                target = feature
        
        current_features.remove(target) ## remove the best feature from the current set of features
        print(f"Feature set {current_features} was best, accuracy is {best_temp_accuracy * 100:.1f}%")

        if best_temp_accuracy > best_accuracy: ## if the current set has better accuracy than the best so far, update
            best_features = current_features.copy()
            best_accuracy = best_temp_accuracy
        else:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n")

    return best_features, best_accuracy

def main():
    print("Welcome to Brandon Tran's Feature Selection Algorithm")
    file_name = input("Type in the name of the file to test: ").strip()
    class_name, features = load_dataset(file_name) ## load the dataset from the specified file

    print ("\nType the number of the algorithm you want to run.")
    print ("1. Forward Selection")
    print ("2. Backward Elimination")
    choice = input("Enter your choice: ").strip()

    cnt_instances = len(class_name) ## the number of instances is the number of rows in the dataset
    cnt_features = len(features[0]) ## the number of features is the number of columns in the dataset (excluding the class name)

    print(f"\nThis dataset has {cnt_features} features (not including the class attribute), with {cnt_instances} instances.")

    all_features = list(range(1, cnt_features + 1)) ## create a list of all feature indices (starting from 1)
    accuracy = leave_one_out(features, class_name, all_features) ## calculate the accuracy using all features
    print(f'Running nearest neighbor with all {cnt_features} features, using "leave-one-out" evaluation, I get an accuracy of {accuracy * 100:.1f}%')

    if choice == '1': ## Choose Forward Selection
        start_time = time.perf_counter()
        best_features, best_accuracy = forward_selection(features, class_name, cnt_features)
        elapsed_time = time.perf_counter() - start_time
            
    elif choice == '2': ## Choose Backward Elimination
        start_time = time.perf_counter()
        best_features, best_accuracy = backward_elimination(features, class_name, cnt_features)
        elapsed_time = time.perf_counter() - start_time
        
    print(f"\nFinished search!! The best feature subset is {best_features}, which has an accuracy of {best_accuracy * 100:.1f}%")
    print(f"Total time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()