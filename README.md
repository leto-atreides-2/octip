# OCTIP: OCT Internship Package

This package was initially developed for my interns working on a dataset of Spectralis OCT images, hence the name.
It contains image preprocessing code.

## Installation

Install TensorFlow [using PIP](https://www.tensorflow.org/install/pip?lang=python3), preferably
using [VirtualEnv](https://www.tensorflow.org/install/pip?lang=python3#2-create-a-virtual-environment-recommended).

> This package was tested successfully with Tensorflow version 2.1.0 and Python version 3.6.8. 

Then, install OCTIP in developer mode (option -e):
```bash
pip install -e octip
```

## Usage Examples

The latter version of OCTIP comes with two scripts:
* `octip-spectralis-2-nifti.py` converts Spectralis OCT dataset in compressed NifTI format,
* `octip-dataset-split.py` splits the converted dataset into train, validation and test subsets.

Let `OCT_Images` denote a directory containing OCT data subdirectories. Follow the instructions below to:
1. convert the dataset with or without retina segmentation.
2. split the dataset into subsets.

### Dataset Conversion Without Retina Segmentation

The following bash command simply converts each eye exam to a volume of 32x224x224 voxels (or more
generally `depth`x`height`x`width` voxels) by:
* selecting 32 B-scans (or more generally `depth` B-scans) per volume,
* resizing each selected B-scan to 224x224 pixels (or more generally `height`x`width` pixels),
* saving the volume as `volumes/patient<x>-eye-exam<y>.nii.gz`, where `<x>` is a unique patient
identifier and `<y>` identifies one exam of one eye of the patient.

```bash
octip-spectralis-2-nifti.py --depth 32 --height 224 --width 224 --input_dirs ../OCT_Images
```
> There can be more than one input directory.

Note that the 32 B-scans are selected as follows:
* if the original volume has less then 32 B-scans, then all B-scans are selected and empty (black)
B-scans are added before and after these B-scans to obtain a total of 32 B-scans.
* if the original volume has 32 B-scans or more, then 32 B-scans are selected uniformly.

The ground truth associated with each volume is summarized in a CSV file called `ground-truth.csv`.
The volume-level ground truth derives from the ground truth associated with the selected B-scans.

### Dataset Conversion With Retina Segmentation

The following bash command converts each eye exam to a volume of 32x192x224 voxels (or more generally
`depth`x`height`x`width` voxels) using retina segmentation:

```bash
CUDA_VISIBLE_DEVICES=0 octip-spectralis-2-nifti.py -r octip_models --depth 32 --height 192 --width 224 \
    --input_dirs ../OCT_Images ../OCT_Non_Interprete
```

Request directory `octip_models` containing the following two model files (gwenole.quellec@inserm.fr):
* `FPN_efficientnetb6_384x384.hdf5`,
* `FPN_efficientnetb7_320x320.hdf5`.

The 32 B-scans (or more generally `depth` B-scans) are selected similarly to the previous example
and the ground truth file is identical. Differences with the previous example are that:
* the retina is segmented in each selected B-scans and only the first 192 pixels
(more generally `height` pixels) below the retinal top surface are selected,
* optionally (if the `--normalize_intensities` option is set), intensity in the selected region is
normalized,
* the selected region is resized to 192x224 pixels (or more generally `height`x`width` pixels).

> Based on initial experiments, the `--normalize_intensities` option seems relevant for progression
> measurement between two exams. However, it does not seem relevant for single exam classification. 

### Viewing the Converted NifTI Volumes

[ImageJ's nifti plugin](https://imagej.nih.gov/ij/plugins/nifti.html) can be used to view the
compressed NifTI files. After installing the plugin, a volume can be imported in ImageJ using
`File -> Import -> NifTI/Analyze`. For convenience, volumes are also stored as directories of PNG
images in the `resized_image` directory.

### Splitting the Converted Dataset into Training, Validation and Test Subsets

The examples above generate a ground truth file called `ground-truth.csv`. To split this file
into training, validation and test subsets, run the following command:

```bash
octip-dataset-split.py -t ground-truth.csv --ratios 0.8 0.1 0.1 \
    --labels MLA DMLA-E DMLA-A OMC-diabetique OMC IVM autres-pathologies
```

This will assign 80% of the patients to the training subset and 10% of them to the validation and
test subsets. More generally, the train:validation:test ratios are given by
`ratios[0]`:`ratios[1]`:`ratios[2]`. This script ensures that:
* the listed labels (MLA, DMLA-E, etc.) are distributed across the three subsets as evenly as possible,
* all exams from the same patient are assigned to the same subset.

Note that this script only splits the ground truth file into subsets (`train.csv`, `validation.csv`
and `test.csv`). The `volumes` directory does not need to be split. Simply use symbolic links if
need be (e.g. for compatibility with my libraries):

```bash
ln -s volumes train
ln -s volumes validation
ln -s volumes test
```
