import os

user = "YOU"


def get_brats_folder(on="val", input_dir="/input"):
    BRATS_VAL_FOLDER = input_dir
    BRATS_TEST_FOLDER = input_dir
    BRATS_TRAIN_FOLDER = input_dir

    if on == "test":
        print(f"Test Input Directory {input_dir}")
        return BRATS_TEST_FOLDER
    elif on == "val":
        print(f"Validation Input Directory {input_dir}")
        return BRATS_VAL_FOLDER
    elif on == "train":
        print(f"Train Input Directory {input_dir}")
        return BRATS_TRAIN_FOLDER
