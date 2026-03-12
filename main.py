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

def main():
    print("Welcome to Brandon's Feature Selection Algorithm")
    file_name = input("Type in the name of the file to test: ").strip()
    class_name, features = load_dataset(file_name) ## load the dataset from the specified file

    cnt_instances = len(class_name) ## the number of instances is the number of rows in the dataset
    cnt_features = len(features[0]) ## the number of features is the number of columns in the dataset (excluding the class name)

    print(f"\nThis dataset has {cnt_features} features (not including the class attribute), with {cnt_instances} instances.")

if __name__ == "__main__":
    main()