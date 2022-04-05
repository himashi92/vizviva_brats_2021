import pathlib

import SimpleITK as sitk
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset
import os
from config import get_brats_folder
from dataset.image_utils import irm_min_max_preprocess, zscore_normalise


class Brats(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, data_aug=False,
                 no_seg=False, normalisation="minmax"):
        super(Brats, self).__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.patterns = ["_t1", "_t1ce", "_t2", "_flair"]
        if not no_seg:
            self.patterns += ["_seg"]
        # for patient_dir in patients_dir:

        files = os.listdir(patients_dir)[0].split("_flair")
        patient_id = files[0]
        paths = [patients_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
        patient = dict(
            id=patient_id, t1=paths[0], t1ce=paths[1],
            t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None
        )
        self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])

        patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
        et_present = 0

        z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
        # Add 1 pixel in each side
        zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
        patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]

        patient_image = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image = [torch.from_numpy(x) for x in patient_image]

        return dict(patient_id=_patient["id"],
                    image=patient_image, seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)


def get_datasets_val(no_seg=True, on="val", normalisation="minmax", input_dir="/input"):

    val_base_folder = pathlib.Path(get_brats_folder(on, input_dir)).resolve()
    assert val_base_folder.exists()
    val_patients_dir = val_base_folder

    bench_dataset = Brats(val_patients_dir, training=False, no_seg=no_seg, benchmarking=True, normalisation=normalisation)

    return bench_dataset




