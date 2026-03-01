import os
import numpy as np
from scipy.io import loadmat

def process_and_save_mat_to_npy(mat_file_path, npy_file_path):
    """
    Load data from a .mat file and convert it to .npy format.

    :param mat_file_path: Path to the input .mat file.
    :param npy_file_path: Path to save the output .npy file.
    """

   
    mat_data = loadmat(mat_file_path)


    data = mat_data['Data']

    data = 10 * np.log10(data)

 
    np.save(npy_file_path, data)
    print(f"Saved {mat_file_path} as {npy_file_path}")

def process_all_mat_files(mat_folder_path, npy_folder_path):
    """
    Iterate through all .mat files in a folder and convert them to .npy format.

    :param mat_folder_path: Path to the folder containing .mat files.
    :param npy_folder_path: Path to the folder where the converted .npy files will be saved.
    """
    os.makedirs(npy_folder_path, exist_ok=True)

    for mat_file in os.listdir(mat_folder_path):
        if mat_file.endswith('.mat'):  
            mat_file_path = os.path.join(mat_folder_path, mat_file)
            npy_file_path = os.path.join(npy_folder_path, mat_file.replace('.mat', '.npy'))

            process_and_save_mat_to_npy(mat_file_path, npy_file_path)


            

# === Set your local data paths here ===
# Update the following paths according to your dataset location

train_mat_path = '2012025_03'          # Directory containing the input .mat files
train_npy_path = 'test_an_npy/image'   # Directory to store the output .npy files

#val_mat_path = 'val'
#val_npy_path = 'val_npy/image'

#test_mat_path = 'test'
#test_npy_path = 'test_npy/image'


process_all_mat_files(train_mat_path, train_npy_path)
#process_all_mat_files(val_mat_path, val_npy_path)

#process_all_mat_files(test_mat_path, test_npy_path)
