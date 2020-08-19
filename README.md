# OCTIP: OCT Internship Package

## Installation

Install TensorFlow [using PIP](https://www.tensorflow.org/install/pip?lang=python3), preferably
using [VirtualEnv](https://www.tensorflow.org/install/pip?lang=python3#2-create-a-virtual-environment-recommended).

> The library was tested successfully with Tensorflow version 2.1.0 and Python version 3.6.8. 

Then, install OCTIP in developer mode (option -e):
```bash
pip install -e octip
```

## Usage Examples

The latter version of OCTIP comes with a script (`octip-convert-2-nifti.py`) that converts Brest's
OCT dataset in compressed NifTI format. First download the `OCT_Images` and `OCT_Non_Interprete`
directories from `synophta3` (195.83.246.82). Then follow the instructions below, to convert the
dataset with or without retina segmentation.

### Without Retina Segmentation

The following bash command simply converts each eye exam to a 32x224x224 voxel volume by:
* selecting 32 B-scans per volume,
* resizing each selected B-scan to a 224x224 pixel image,
* saving the volume in `volume/patient<x>-eye-exam<y>.nii.gz`, where `<x>` is a unique patient
identifier and `<y>` identifies one exam of one eye of the patient.

```bash
octip-convert-2-nifti.py --depth 32 --height 224 --width 224 --input_dirs ../OCT_Images ../OCT_Non_Interprete
```

Note that the 32 B-scans are selected as follows:
* if the original volume has less then 32 B-scans, then all B-scans are selected and empty (black)
B-scans are added before and after these B-scans to obtain a total of 32 B-scans.
* if the original volume has 32 B-scans or more, then 32 B-scans are selected uniformly.

Of course, the volume's output size can be changed.

The ground truth associated with each volume is summarized in a CSV file called `ground-truth.csv`.
The volume-level ground truth derives from the ground truth associated with the selected B-scans.

### Without Retina Segmentation

The following bash command converts each eye exam to a 32x192x224 voxel volume using retina
segmentation and, optionally, intensity normalization:

```bash
CUDA_VISIBLE_DEVICES=0 octip-convert-2-nifti.py -r octip_models --depth 32 --height 192 --width 224 --input_dirs ../OCT_Images ../OCT_Non_Interprete
```

Directory `octip_models` should contain the following two model files, downloaded from my Dropbox:
* `FPN_efficientnetb6_384x384.hdf5`,
* `FPN_efficientnetb7_320x320.hdf5`.

The 32 B-scans are selected similarly to the previous example and the ground truth file is
identical. Differences with the previous example are that:
* the retina is segmented in each selected B-scans and only the first 192 pixels below the retinal
top surface are selected,
* optionally (if the `--native_intensities` option is used), intensity in the selected region is
normalized,
* the selected region is resized to a 192x224 pixel.

### Viewing the Converted NifTI Volumes

[ImageJ's nifti plugin](https://imagej.nih.gov/ij/plugins/nifti.html) can be used to view the
compressed NifTI files. After installing the plugin, a volume can be imported in ImageJ using
`File -> Import -> NifTI/Analyze`. For convenience, volumes are also stored as directories of PNG
images in the `resized_image` directory.

## Editing Instructions

* If you add or edit a Python file, please add yourself as an author of this file.
* Please comment your code sufficiently and rigorously.
* Do not duplicate code.
* Etc. In a word, be a nice coder ;-)
