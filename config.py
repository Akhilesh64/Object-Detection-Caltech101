import os

BASE_PATH = os.path.join(os.getcwd(), 'dataset')
IMAGES_PATH = os.path.join(BASE_PATH, 'images')
ANNOTS_PATH = os.path.join(BASE_PATH, 'annotations_csv')

BASE_OUTPUT = os.path.join(os.getcwd(), 'output')
MODEL_PATH = os.path.join(BASE_OUTPUT, 'detector.h5')
LB_PATH = os.path.join(BASE_OUTPUT, 'lb.pickle')
TEST_PATHS = os.path.join(BASE_OUTPUT, 'test_paths.txt')

LR = 0.0001
NUM_EPOCHS = 25
BATCH_SIZE = 4

