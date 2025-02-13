import os
import random
import shutil

def copy_random_mp3_samples(source_dir, target_dir, sample_count=300):
    """Copies a random sample of mp3 files from source_dir to target_dir."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    mp3_files = []
    
    # Walk through the directory and collect all mp3 files
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.flac'):
                mp3_files.append(os.path.join(root, file))

    # Randomly select samples
    selected_samples = random.sample(mp3_files, min(sample_count, len(mp3_files)))

    # Copy selected samples to the target directory
    for sample in selected_samples:
        shutil.copy(sample, target_dir)

# Usage
copy_random_mp3_samples(os.getcwd(), 'dataset')
