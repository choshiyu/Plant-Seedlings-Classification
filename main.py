import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import model_building
from keras.callbacks import History

history = History()

epoch_num = 40
batch_size = 16
width = 299
height = 299
label_list = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen",
                "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
                "Sugar beet"]
# model_list = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB4', 'EfficientNetB0',
#             'MobileNetV3Small', 'InceptionResNetV2']

model_list = ['EfficientNetB4', 'EfficientNetB0',
            'MobileNetV3Small', 'InceptionResNetV2']
if __name__ == '__main__':
    
    for base_model in model_list:
    
        model = model_building.define_model(width, height, base_model)
        model.summary()
        train_generator, validation_generator, test_generator = model_building.define_generators(width, height, batch_size)
        
        save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./model.h5',
        monitor='val_acc',
        save_best_only=True,
        verbose=1)
        
        model.fit(
            train_generator, # train
            epochs = epoch_num,
            steps_per_epoch = train_generator.samples // batch_size, # train steps
            validation_data = validation_generator, # val
            validation_steps = validation_generator.samples // batch_size, # validation steps
            callbacks=[history]
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(history.history['accuracy'], color='blue')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')

        ax2.plot(history.history['loss'], color='orange')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        plt.tight_layout()
        plt.show()

        predictions = model.predict(test_generator, steps=test_generator.samples)

        class_list = []
        for i in range(0, predictions.shape[0]):
            y_class = predictions[i, :].argmax(axis=-1)
            class_list += [label_list[y_class]]

        submission = pd.DataFrame()
        submission['file'] = test_generator.filenames
        submission['file'] = submission['file'].str.replace('test\\\\', '')
        submission['species'] = class_list
        submission.to_csv(f'submission{base_model}.csv', index=False)