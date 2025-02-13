import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf

def convert_h5_to_tflite(h5_path, tflite_path):
    # Load the Keras model
    model = tf.keras.models.load_model(h5_path)

    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python h5_to_tflite.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    tflite_path = file_path.replace('.h5', '.tflite')
    convert_h5_to_tflite(file_path, tflite_path)
    print(f"Converted {file_path} to {tflite_path}")