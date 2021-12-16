import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_paths(path):
    train_csv = pd.read_csv('data/train.csv')
    # Prepend image filenames in train/ with relative path
    filenames = ['train/' + fname for fname in train_csv['id'].tolist()]
    labels = train_csv['has_cactus'].tolist()
    train_filenames, val_filenames, train_labels, val_labels = train_test_split(filenames,labels,train_size=0.9,random_state=42)
    return train_filenames, val_filenames, train_labels, val_labels


def get_dataset(file_names, labels):
    data_set = tf.data.Dataset.from_tensor_slices(
    (tf.constant(file_names), tf.constant(labels)))
    return data_set

IMAGE_SIZE = 224 # Minimum image size for use with MobileNetV2
BATCH_SIZE = 32
learning_rate = 0.0001
# Function to load and preprocess each image

def parse_image(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img)
    img = (tf.cast(img, tf.float32)/127.5) - 1
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, label

def build_mobilenet():
    IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
    # Pre-trained model with MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the pre-trained model weights
    base_model.trainable = False
    # Trainable classification head
    maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')# Layer classification head with feature detector
    model = tf.keras.Sequential([
        base_model,
        maxpool_layer,
        prediction_layer
    ])
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
                loss='binary_crossentropy',
                metrics=['accuracy']
    )
    return base_model, model

def train_model(model, base_model, train_data, val_data):
    num_epochs = 30
    steps_per_epoch = round(len(train_data))//BATCH_SIZE
    val_steps = 20
    model.fit(train_data.repeat(),
            epochs=num_epochs,
            steps_per_epoch = steps_per_epoch,
            validation_data=val_data.repeat(), 
            validation_steps=val_steps)


    base_model.trainable = True

    # Refreeze layers until the layers we want to fine-tune
    for layer in base_model.layers[:100]:
        layer.trainable =  False
        # Use a lower learning rate
        lr_finetune = learning_rate / 10
        # Recompile the model
        model.compile(loss='binary_crossentropy',
                    optimizer = tf.keras.optimizers.Adam(lr=lr_finetune),
                    metrics=['accuracy'])# Increase training epochs for fine-tuning
        fine_tune_epochs = 30
        total_epochs =  num_epochs + fine_tune_epochs
        # Fine-tune model
        # Note: Set initial_epoch to begin training after epoch 30 since we
        # previously trained for 30 epochs.
        model.fit(train_data.repeat(), 
                steps_per_epoch = steps_per_epoch,
                epochs=total_epochs, 
                initial_epoch = num_epochs,
                validation_data=val_data.repeat(), 
                validation_steps=val_steps)
    
    return model