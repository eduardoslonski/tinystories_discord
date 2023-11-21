import pickle
import numpy as np

def gather_tokenized(input_file, output_file):
    with open(input_file, 'rb') as infile:
        loaded_tokenized_data = pickle.load(infile)
        
    loaded_tokenized_data_array = [np.array(lst, dtype=np.uint16) for lst in loaded_tokenized_data]
    concatenated_data = np.concatenate(loaded_tokenized_data_array)

    with open(output_file, 'wb') as outfile:
        concatenated_data.tofile(outfile)

if __name__ == "__main__":
    gather_tokenized('data/train_tokenized.pkl', 'data/train_together_tokenized.bin')