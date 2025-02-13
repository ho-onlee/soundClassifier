import os, sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import json
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient
import matplotlib.pyplot as plt
from urllib.parse import unquote
from tqdm import tqdm
import h5py, requests
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
# os.chdir(r'C:\Users\Hoon\Nextcloud3\Projects\SoundTesting')

def grabAPI():
    token = '03b886f9c8850386fb40064cea4042474b7ce2b0'
    project_id = '1'  # Replace with the actual project ID
    url = f'http://192.168.2.5:8080/api/projects/{project_id}/export?exportType=JSON&download_all_tasks=true'
    headers = {
        'Authorization': f'Token {token}'
    }
    response = requests.get(url, headers =headers, verify=True)  # Set verify=False to ignore SSL warnings

    if response.status_code == 200:
        with open('exported_data.json', 'w', encoding='utf-8') as json_file:
            json_file.write(response.text)
        print("Data exported successfully.")
        return True
    else:
        print(f"Failed to export data. Status code: {response.status_code}")
        return False


def grabAudio(filename:str)->bool:
    origin_d = '/home/worker/.local/share/label-studio/media/upload/1/'+unquote(filename)
    target_d = os.path.join(os.getcwd(), 'dataset')
    if os.path.exists(os.path.join(target_d, filename)): 
        return os.path.join(target_d, filename)
    # print(f"Retrieving File: {origin_d} -> {target_d}")
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.load_system_host_keys()
    client.connect('192.168.2.5', username='worker', password='worker')

    scp = SCPClient(client.get_transport())
    scp.get(origin_d, target_d)    
    if os.path.exists(os.path.join(target_d, filename)): 
        return os.path.join(target_d, filename)
    print("Something went Wrong.")
    return False

def check_file(filename:str):
    target_d = os.path.join(os.getcwd(), 'dataset', filename)
    return os.path.exists(target_d)


def split_dataset(dataset, chunk_size=32000*1):
    """Split the dataset into chunks of specified size."""
    hop_size = chunk_size // 2  # Define hop size (e.g., half of chunk_size)
    split_data = {}
    for label, audio_waveform in dataset.items():
        split_data[label] = [
            audio_waveform[i:i + chunk_size] 
            for i in range(0, len(audio_waveform) - chunk_size + 1, hop_size)  # Use hop_size for stepping
        ]
    return split_data


def prepare_data():
    if not grabAPI(): raise("Not able to retrieve JSON")
    if not os.path.exists('dataset'): os.mkdir('dataset')
    if os.path.exists('split_dataset.npy'): return np.load('split_dataset.npy', allow_pickle=True).item()
    with open('exported_data.json', 'r', encoding='utf-8') as json_file:  # Specify encoding
        js = json.load(json_file)  # Load the JSON data
    dataset = dict()
    for entry in tqdm(js, total=len(js)):
        try:
            filename = entry["file_upload"]
            r_annotations = entry['annotations']
            annotations = [[e['result'][0]['value']['labels'][0],float(e['result'][0]['value']['start']),float(e['result'][0]['value']['end'])] for e in r_annotations ]                  
            path = grabAudio(filename)
            audio_waveform, sample_rate = librosa.load(path, sr=32000)
            for label, start, end in annotations:
                if label == 'Composition': continue
                if label == 'drawers': 'Cabinet'
                if label == 'Rolling Carts': label = 'HVAC'
                if label == 'Sink/Water': label = 'medical air valve'
                if label in ['ICU Medical', 'Baxter', 'Alaris', 'alarm', 'siren', 'SpaceLAbs']: label = 'Hospital Devices'
                if label not in dataset.keys(): dataset[label] = []
                start = int(start*sample_rate)
                end = int(end*sample_rate)
                if end-start < 32000:
                    np.pad(audio_waveform, (0, 32000), 'constant')
                dataset[label] += list(audio_waveform[start:end])
        except Exception as e:
            print(filename)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            continue
    speech_dataset_path = os.path.join(os.getcwd(), 'speech_dataset')
    for root, _, files in os.walk(speech_dataset_path):
        for file in files[:100]:
            if file.endswith('.flac'):  # Assuming the audio files are in .wav format
                label = 'Speech'  # Label for the audio files
                path = os.path.join(root, file)
                audio_waveform, sample_rate = librosa.load(path, sr=32000)
                if label not in dataset.keys():
                    dataset[label] = []
                dataset[label] += list(audio_waveform)
    splited_dataset = split_dataset(dataset)
    np.save('split_dataset.npy', splited_dataset)
    return np.load('split_dataset.npy', allow_pickle=True).item()

def get_mfcc(splited_dataset):
    if os.path.exists('mfcc_features.npy'): return np.load('mfcc_features.npy', allow_pickle=True).item()
    mfcc_features = dict()
    
    for label, audio_chunks in tqdm(splited_dataset.items(), total=len(splited_dataset), desc='get_mfcc'):
        mfcc_features[label] = []
        for audio_waveform in audio_chunks:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=np.array(audio_waveform), sr=32000, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            mfcc_features[label].append(mfccs_scaled)
    np.save('mfcc_features.npy', mfcc_features)
    return mfcc_features

def split_data(dataset):
    x = []
    y = []
    labels = []  # Create a label array
    for label, features in dataset.items():
        for feature in features:
            x.append(feature)
            if label not in labels:  # Check if the label already exists
                labels.append(label)  # Add the label if it does not exist
            y.append(labels.index(label))  # Set y as the corresponding index of that label
    
    x = np.array(x)
    y = to_categorical(np.array(y))
    
    # Optionally, you can split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Print the lengths of the datasets
    print(f"Training set size: {len(x_train)}")
    print(f"Testing set size: {len(x_test)}")
    
    return x_train, x_test, y_train, y_test, labels

def evaluate(model, x_test, y_test, label):
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Make predictions on the test set
    predictions = model.predict(x_test)

    # Convert predictions from probabilities to class labels
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Generate a classification report
    from sklearn.metrics import classification_report
    report = classification_report(true_classes, predicted_classes, target_names=[label[i] for i in np.unique(true_classes)])
    print(report)

    # Plot a few test samples and their predicted labels
    num_samples = len(true_classes)
    plt.figure(figsize=(15, 5))
    # for i in range(num_samples):
    #     plt.subplot(1, num_samples, i + 1)
    #     plt.plot(x_test[i])
    plt.bar(range(num_samples), true_classes, color='blue', alpha=0.5)
    plt.title(f'Predicted: {label[predicted_classes[i]]}')
    plt.axis('off')
    plt.show()


def build_model(x_train, x_test, y_train, y_test):
    input_shape = (x_train.shape[1], 1)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(y_train[0]), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file="wfmodel.png", show_shapes=True, show_layer_activations=True)
    return model

def train(model,x_train, x_test, y_train, y_test ):
    callback = EarlyStopping(patience=5, min_delta=1e-4, monitor='accuracy')
    history1 = model.fit(x_train, 
            y_train, 
            batch_size=8,
            epochs=100000, 
            validation_data=(x_test, y_test), 
            verbose=1, 
            callbacks=[callback,]
            )
    # fig=plt.figure(figsize=(12,4))
    plt.plot(history1.epoch, history1.history['loss'], label="Dense")
    # plt.plot(history1.epoch, history1.history['accuracy'], label="Dense")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def save_model_ext(model, filepath, overwrite=True, meta_data=None):
    tf.keras.models.save_model(model, filepath, overwrite)
    if meta_data is not None:
        f = h5py.File(filepath, mode='a')
        f.attrs['label_data'] = meta_data
        f.close()

def main():
    splited_dataset = prepare_data()    
    mfcc_features = get_mfcc(splited_dataset)
    x_train, x_test, y_train, y_test, label = split_data(mfcc_features)
    model = build_model(x_train, x_test, y_train, y_test)
    train(model, x_train, x_test, y_train, y_test)

    # Visualization
    # print("Shapes of MFCC features (sorted by decreasing order):")
    # sorted_features = sorted(mfcc_features.items(), key=lambda x: len(x[1]), reverse=True)
    # for label, features in sorted_features:
    #     print(f"{label}: {np.array(features).shape}")

if __name__ == "__main__":
    main()

