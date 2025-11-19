import os
import re
import shutil
import json
from pathlib import Path


class FormatConverter:
    def __init__(self, src_inputs, src_targets, dst_root):
        self.src_inputs = Path(src_inputs)
        self.src_targets = Path(src_targets)
        self.dst_root = Path(dst_root)
        self.img_dir = self.dst_root / "images"
        self._prepare_dirs()

    def _prepare_dirs(self):
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def convert(self):
        for file in self.src_inputs.iterdir():
            name = file.name

            if name.endswith(".csv"):
                continue

            if name.endswith(".json"):
                self._process_json(file)

            if name.endswith(".nii.gz") and "img" in name:
                self._process_img(file)

    def _process_json(self, file):
        fname = file.stem
        m = re.match(r"([a-z0-9]+)_(BL|FU)_\d+", fname)
        if not m:
            return
        pid, phase = m.groups()
        label = "baseline" if phase == "BL" else "followup"
        if "BL" in fname:
            prefix = "primary"
        else:
            prefix = "secondary"

        new_name = f"{prefix}-{label}-lesion-clicks.json"
        shutil.copy(file, self.dst_root / new_name)

    def _process_img(self, file):
        fname = file.stem
        m = re.match(r"([a-z0-9]+)_(BL|FU)_img_.*", fname)
        if not m:
            return
        pid, phase = m.groups()
        label = "baseline" if phase == "BL" else "followup"
        if "BL" in fname:
            prefix = "primary"
        else:
            prefix = "secondary"

        new_name = f"{prefix}-{label}-ct.nii.gz"
        shutil.copy(file, self.img_dir / new_name)

        self._process_mask(pid, prefix, label)

    def _process_mask(self, pid, prefix, label):
        pattern = re.compile(rf"{pid}_.*mask_.*\.nii\.gz")
        for f in self.src_inputs.iterdir():
            if pattern.match(f.name):
                new_name = f"{prefix}-{label}-ct-tumor-lesion-seg.nii.gz"
                shutil.copy(f, self.img_dir / new_name)
                return


if __name__ == "__main__":
    converter = FormatConverter(
        src_inputs="/data/t/Rory/Deep Learning/2.Data/Longitudinal-CT/nnUNet_raw/Dataset001_LongitudinalCT/inputsTr",
        src_targets="/data/t/Rory/Deep Learning/2.Data/Longitudinal-CT/nnUNet_raw/Dataset001_LongitudinalCT/targetsTr",
        dst_root="/data/t/Deep Learning/2.Data/Longitudinal-CT/reformatted_data_for_mast_baselinet"
    )
    converter.convert()
