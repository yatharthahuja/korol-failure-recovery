import pickle

# Path to the pickle file
file_path = './demo1/joint_states.pickle'

# Open and read the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Print the data
print(data[0])
print(type(data[0]))