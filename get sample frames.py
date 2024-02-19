import os
import random

def randomly_delete_files(folder_path, num_files_to_keep=200):
    # Get list of files in the folder
    files = os.listdir(folder_path)
    
    # Calculate number of files to delete
    num_files_to_delete = len(files) - num_files_to_keep

    # If there are more files than the specified number to keep, delete some randomly
    if num_files_to_delete > 0:
        files_to_delete = random.sample(files, num_files_to_delete)
        for file_to_delete in files_to_delete:
            file_path = os.path.join(folder_path, file_to_delete)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

if __name__ == "__main__":
    folder_path = 'output'
    randomly_delete_files(folder_path)
