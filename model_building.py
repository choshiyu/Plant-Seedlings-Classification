import tensorflow as tf
from keras.callbacks import History
history = History()

def define_model(width, height, model):

    model_input = tf.keras.layers.Input(shape=(width, height, 3), name='image_input')
    
    if model == 'InceptionResNetV2':
        model_main = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet')(model_input)
    elif model == 'DenseNet121':
        model_main = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet')(model_input)
    elif model == 'DenseNet169':
        model_main = tf.keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet')(model_input)
    elif model == 'DenseNet201':
        model_main = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet')(model_input)
    elif model == 'EfficientNetB4':
        model_main = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, weights='imagenet')(model_input)  
    elif model == 'EfficientNetB0':
        model_main = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet')(model_input)  
    elif model == 'MobileNetV3Small':
        model_main = tf.keras.applications.MobileNetV3Small(include_top=False, weights='imagenet')(model_input)
    elif model == 'restnet50':
        model_main = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')(model_input)
    elif model == 'restnet101':
        model_main = tf.keras.applications.ResNet101(include_top=False, weights='imagenet')(model_input)

    model_dense1 = tf.keras.layers.Flatten()(model_main) # Flatten
    model_dense2 = tf.keras.layers.Dense(128, activation='relu')(model_dense1)
    model_out = tf.keras.layers.Dense(12, activation="softmax")(model_dense2)

    model = tf.keras.models.Model(model_input, model_out)
    optimizer = tf.keras.optimizers.Adam(lr=0.00004, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    return model

def define_generators(width, height, batch_size):

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.5,
        vertical_flip=True,
        horizontal_flip=True,
        validation_split=0.0) # 以整個訓練集進行訓練

    # train
    train_generator = train_datagen.flow_from_directory(
    directory='./train',
    target_size=(width, height),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode="categorical", # one-hot的類別向量
    subset='training') # 指定使用訓練集
    
    # val
    validation_generator = train_datagen.flow_from_directory(
        directory='./train',
        target_size=(width, height),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode="categorical",
        subset='validation')

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        directory='./',
        classes=['test'],
        target_size=(width, height),
        batch_size=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical')

    return train_generator, validation_generator, test_generator