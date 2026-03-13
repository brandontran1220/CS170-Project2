def load_dataset(filename):
    features = []
    class_name = []

    with open(filename, 'r') as file: ## open the file for reading
        for line in file:
            line = line.strip()
            if not line:
                continue
            
            nums = [float(x) for x in line.split()] ## split the line into numbers and convert them to float

            class_name.append(nums[0])  ## the class name is the first number in the row
            features.append(nums[1:])  ## the rest of the numbers are the features
    
    return class_name, features

def euclidean_distance(first, second, feature_set):
    distance = 0.0
    for i in feature_set: ## iterate through the specified features
        distance += (first[i - 1] - second[i - 1]) ** 2 ## calculate the squared difference for each feature and sum them up
    return distance ** 0.5 ## return the Euclidean distance

def leave_one_out(features, class_name, feature_set):
    num_correct = 0

    for i in range(len(features)): ## iterate through each instance in the dataset
        nearest_neighbor = None
        nearest_distance = float('inf')

        for j in range(len(features)): ## iterate through the dataset again to find the nearest neighbor
            if i == j:
                continue
            
            distance = euclidean_distance(features[i], features[j], feature_set) ## calculate the distance between the two features

            if distance < nearest_distance: ## if this instance is closer than the previous nearest neighbor, update the nearest neighbor and distance
                nearest_neighbor = j
                nearest_distance = distance
        
        if class_name[i] == class_name[nearest_neighbor]: ## Increment number of correct classifications if the nearest neighbor has the same class name as the current instance
            num_correct += 1

    return num_correct / len(features) ## return the accuracy as the percentage of correct classifications 

def main():
    print("Welcome to Brandon's Feature Selection Algorithm")
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

if __name__ == "__main__":
    main()