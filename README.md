# OCTIP: OCT Internship Package

## Installation

Install TensorFlow [using PIP](https://www.tensorflow.org/install/pip?lang=python3), preferably using [VirtualEnv](https://www.tensorflow.org/install/pip?lang=python3#2-create-a-virtual-environment-recommended).

Then, install OCTIP in developer mode (option -e):
```bash
pip install -e octip
```

## Usage Example

In the following example, an OCT volume in XML format is parsed and preprocessed: 

```python
import octip

model_directory = 'octip_models'
segmentation_directories = ['octip_data/segmentations1', 'octip_data/segmentations2']
output_directory = 'octip_data/preprocessed'

# parsing the XML file
bscans = octip.XMLParser('octip_data/example/E70C7490.xml', False).sorted_bscans()

# segmenting the retina in all B-scans with the first model 
localizer1 = octip.RetinaLocalizer('FPN', 'efficientnetb6', (384, 384),
                                   model_directory = model_directory)
localizer1(octip.RetinaLocalizationDataset(bscans, 8, localizer1),
           segmentation_directories[0])

# segmenting the retina in all B-scans with the second model
localizer2 = octip.RetinaLocalizer('FPN', 'efficientnetb7', (320, 320),
                                   model_directory = model_directory)
localizer2(octip.RetinaLocalizationDataset(bscans, 8, localizer2),
           segmentation_directories[1])

# pre-processing the B-scans
preprocessor = octip.PreProcessor(200, min_height = 100, normalize_intensities = True)
preprocessor(bscans, segmentation_directories, output_directory)

# forming the C-scan
cscan = octip.bscans_to_cscan(bscans, output_directory, '.png')
print(cscan.shape)
```

## Editing Instructions

* If you add or edit a Python file, please add yourself as an author of this file.
* Please comment your code sufficiently and rigorously.
* Do not duplicate code.
* Etc. In a word, be a nice coder ;-)
