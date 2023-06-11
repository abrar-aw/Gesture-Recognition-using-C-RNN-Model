from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import TimeDistributed, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, GRU


def create_model_BDslr(frames, width, height, channels, output):
    """
    Create the keras model.

    :param frames: frame number of the sequence.
    :param width: width of the image.
    :param height: height of the image.
    :param channels: 3 for RGB, 1 for B/W images.
    :param output: number of neurons for classification.
    :return: the keras model.
    """
    model = Sequential([
        # ConvNet
        TimeDistributed(
            MobileNetV2(weights='imagenet', include_top=False,
                        input_shape=[height, width, channels]),
            input_shape=[frames, height, width, channels]
        ),
        TimeDistributed(GlobalAveragePooling2D()),


        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Bidirectional(LSTM(128)),

        # Feedforward
        Dense(units=200, activation='relu'),
        Dropout(0.5),
        Dense(units=150, activation='relu'),
        Dropout(0.5),
        Dense(units=output, activation='softmax')
    ])

    return model
