import os

DATA_PATH = "../data"
TRAIN_PATH = os.path.join(DATA_PATH, "training_images")
TEST_PATH = os.path.join(DATA_PATH, "testing_images")
TRAIN_CSV = "train_solution_bounding_boxes (1).csv"

ARCHITECTURE = "tf_efficientnetv2_s"