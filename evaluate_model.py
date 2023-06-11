from tensorflow.keras.models import load_model
from data_utils import labels_to_number, videos_to_dict
from frame_generator import VideoFrameGenerator
from models import create_model_BDslr


# model settings
height = 224
width = 224
dim = (height, width)
batch_size = 8
frames = 10
channels = 3
output = 200

TEST_PATH = '.\\data\\test\\'
labels = labels_to_number(TEST_PATH)
y_test_dict = videos_to_dict(TEST_PATH, labels)
X_test = list(y_test_dict.keys())

last = load_model(r'E:\slproject\saved_models\200w_model_v5.h5')

model = create_model_BDslr(frames, width, height, channels, output)
model.compile(optimizer=last.optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.set_weights(last.get_weights())

print('\nTest generator')
test_generator = VideoFrameGenerator(
    list_IDs=X_test,
    labels=y_test_dict,
    batch_size=batch_size,
    dim=dim,
    n_channels=channels,
    n_sequence=frames,
    shuffle=False,
    type_gen='test'
)

print('\nEvaluating the best model on test set . . .')
eval_loss, eval_acc = model.evaluate(test_generator)
