import os
import random
import string

def generate_random_phone_number():
    return ''.join(random.choices(string.digits, k=10))

def rename_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            new_name = generate_random_phone_number() + '.txt'
            new_path = os.path.join(directory, new_name)
            
            # If the file name exists, append a random number
            while os.path.exists(new_path):
                random_suffix = ''.join(random.choices(string.digits, k=3))
                new_name = generate_random_phone_number() + '_' + random_suffix + '.txt'
                new_path = os.path.join(directory, new_name)
            
            old_path = os.path.join(directory, filename)
            os.rename(old_path, new_path)
            print(f'Renamed {old_path} to {new_path}')

# Usage
sms_directory = './sms-data'  # Replace with the actual directory
rename_files_in_directory(sms_directory)