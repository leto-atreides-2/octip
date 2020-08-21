"""
OCTIP script for converting Brest's OCT dataset in compressed NifTI format.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'Proprietary'
__version__ = '1.1'

import cv2
import glob
import nibabel as nib
import numpy as np
import octip
import os
import pandas as pd
import sys
from argparse import ArgumentParser
from collections import defaultdict

labels = ['Drusens', 'AEP', 'DSR', 'DEP', 'SHE', 'AG', 'Logettes', 'AER', 'Exsudats', 'MER', 'TM',
          'TVM', 'Autres Lésions', 'MLA', 'DMLA E', 'DMLA A', 'OMC diabètique', 'OMC', 'IVM',
          'Autres patho']
clean_labels = ['drusen', 'AEP', 'DSR', 'DEP', 'SHE', 'AG', 'logettes', 'AER', 'exsudats', 'MER',
                'TM', 'TVM', 'autres-lesions', 'MLA', 'DMLA-E', 'DMLA-A', 'OMC-diabetique', 'OMC',
                'IVM', 'autres-pathologies']


def resize_images(args,
                  selected_base_names,
                  input_dir,
                  output_dir,
                  change_ext = False):
    """
    Resizes all B-scans in a volume.

    :param args: command-line arguments
    :param selected_base_names: base name of the selected B-scans
    :param input_dir: directory containing the original B-scans
    :param output_dir: directory containing the resized B-scans
    :param change_ext: whether the extension of input image files should be changed to .png

    :return: the list of resized B-scans
    """
    volume = []
    for url, i in zip(selected_base_names, range(args.depth)):
        if url is not None:
            if change_ext:
                url_ext = os.path.splitext(url)[0] + '.png'
            else:
                url_ext = url
            image = cv2.imread(os.path.join(input_dir, url_ext), cv2.IMREAD_GRAYSCALE)
            if image.shape != (args.height, args.width):
                image = cv2.resize(image, (args.width, args.height))
        else:
            image = np.zeros((args.height, args.width), np.uint8)
        volume.append(image)
        cv2.imwrite(os.path.join(output_dir, '{}.png'.format(i)), image)
    return volume


def main():
    """
    Converts Brest's OCT dataset in compressed NifTI format.
    """

    # parsing the command line
    parser = ArgumentParser(
        description = 'Converts Brest\'s OCT dataset in compressed NifTI format.')
    parser.add_argument('-i', '--image_dir', default = 'resized_images',
                        help = 'directory containing selected and resized 2-D images '
                               '(can be safely removed at the end of this script)')
    parser.add_argument('-v', '--volume_dir', default = 'volumes',
                        help = 'directory containing volumes in compressed NifTI format')
    parser.add_argument('-g', '--ground_truth', default = 'ground-truth.csv',
                        help = 'CSV file containing the labels')
    parser.add_argument('--depth', type = int, default = 32,
                        help = 'number of images selected per volume')
    parser.add_argument('--height', type = int, default = 496, help = 'image height after resizing')
    parser.add_argument('--width', type = int, default = 512, help = 'image width after resizing')
    parser.add_argument('-r', '--retina_model_dir', default = None,
                        help = 'directory containing retina segmentation models')
    parser.add_argument('--input_dirs', required = True, nargs = '+',
                        help = 'space-delimited list of input directories')
    parser.add_argument('--normalize_intensities', dest = 'normalize_intensities',
                        action = 'store_true',
                        help = 'if a retina segmentation model is provided, '
                               'intensities should be normalized')
    parser.add_argument('--native_intensities', dest = 'normalize_intensities',
                        action = 'store_false',
                        help = 'even if a retina segmentation model is provided, '
                               'intensities should not be normalized')
    parser.set_defaults(normalize_intensities = False)
    if len(sys.argv[1:]) == 0:
        parser.print_usage()
        parser.exit()
    args = parser.parse_args()

    # output volume directory
    if not os.path.exists(args.volume_dir):
        os.makedirs(args.volume_dir)

    # preparing retina segmentation if requested
    if args.retina_model_dir is not None:
        localizer1 = octip.RetinaLocalizer('FPN', 'efficientnetb6', (384, 384),
                                           model_directory = args.retina_model_dir)
        localizer2 = octip.RetinaLocalizer('FPN', 'efficientnetb7', (320, 320),
                                           model_directory = args.retina_model_dir)
    else:
        localizer1, localizer2 = None, None

    # loop over patients
    patient_id, num_exams = 0, 0
    num_labels = defaultdict(int)
    with open(args.ground_truth, 'w') as ground_truth:
        ground_truth.write(','.join(['urls', 'normal'] + clean_labels) + '\n')
        for main_dir in args.input_dirs:
            for patient in glob.glob(os.path.join(main_dir, '*')):
                excel_files = glob.glob(os.path.join(patient, '**', '*.xlsx'), recursive = True)
                if len(excel_files) == 0:
                    continue
                print('Processing patient {}...'.format(patient))

                # loop over eye exams
                eye_exam_id = 0
                for excel_file in excel_files:
                    eye_exam = os.path.split(excel_file)[0]
                    volume_name = 'patient{}-eye-exam{}'.format(patient_id, eye_exam_id)

                    # selecting images
                    images = glob.glob(os.path.join(eye_exam, '*.bmp'))
                    base_names = [os.path.basename(image) for image in images]
                    image_ids = [int(base_name.split('.')[0]) for base_name in base_names]
                    base_names = [x for _, x in sorted(zip(image_ids, base_names))]
                    if base_names[0] == '0.bmp':
                        base_names = base_names[1:]
                    if len(base_names) == 0:
                        continue
                    if len(base_names) < args.depth:
                        delta = args.depth - len(base_names)
                        left = delta // 2
                        right = delta - left
                        selected_base_names = [None] * left + base_names + [None] * right
                    elif len(base_names) > args.depth:
                        selected_base_names = []
                        for i in range(args.depth):
                            proportion = float(i) / (args.depth - 1)
                            idx = round(proportion * (len(base_names) - 1))
                            selected_base_names.append(base_names[idx])
                    else:
                        selected_base_names = base_names

                    # converting annotations
                    data = pd.read_excel(excel_file)
                    urls = data['NomImage'].values.tolist()
                    normal = True
                    diagnoses = []
                    for label, clean_label in zip(labels, clean_labels):
                        bscan_labels = dict(zip(urls, data[label].values.tolist()))
                        positive = False
                        for url in selected_base_names:
                            if url is not None:
                                diagnosis = bscan_labels[url]
                                if diagnosis:
                                    positive = True
                                    break
                        if positive:
                            num_labels[clean_label] += 1
                            normal = False
                            diagnoses.append('1')
                        else:
                            diagnoses.append('0')
                    if normal:
                        num_labels['normal'] += 1
                    ground_truth.write(','.join([volume_name + '.nii.gz',
                                                 '1' if normal else '0'] + diagnoses) + '\n')

                    # resizing images
                    exam_image_dir = os.path.join(args.image_dir, volume_name)
                    if not os.path.exists(exam_image_dir):
                        os.makedirs(exam_image_dir)
                    if args.retina_model_dir is not None:

                        # selecting non-empty images
                        bscans = []
                        for url in selected_base_names:
                            if url is not None:
                                bscans.append(os.path.join(eye_exam, url))

                        # segmenting and preprocessing images
                        segmentation_directory_1 = os.path.join(exam_image_dir, 'segmentation1')
                        segmentation_directory_2 = os.path.join(exam_image_dir, 'segmentation2')
                        preprocessed_dir = os.path.join(exam_image_dir, 'preprocessed')
                        if not os.path.exists(segmentation_directory_1):
                            os.makedirs(segmentation_directory_1)
                        if not os.path.exists(segmentation_directory_2):
                            os.makedirs(segmentation_directory_2)
                        if not os.path.exists(preprocessed_dir):
                            os.makedirs(preprocessed_dir)
                        localizer1(octip.RetinaLocalizationDataset(bscans, 4, localizer1),
                                   segmentation_directory_1)
                        localizer2(octip.RetinaLocalizationDataset(bscans, 4, localizer2),
                                   segmentation_directory_2)
                        preprocessor = octip.PreProcessor(
                            args.height, min_height = 100,
                            normalize_intensities = args.normalize_intensities)
                        preprocessor(bscans, [segmentation_directory_1, segmentation_directory_2],
                                     preprocessed_dir)

                        # resizing preprocessed images
                        volume = resize_images(args, selected_base_names, preprocessed_dir,
                                               exam_image_dir, True)

                    else:

                        # resizing images
                        volume = resize_images(args, selected_base_names, eye_exam, exam_image_dir)

                    volume = np.asarray(volume)

                    # saving the volume
                    img = nib.Nifti1Image(volume, np.eye(4))
                    nib.save(img, os.path.join(args.volume_dir, volume_name + '.nii.gz'))

                    eye_exam_id += 1
                    num_exams += 1
                if eye_exam_id > 0:
                    patient_id += 1

    # summary
    with open('conversion-summary.yml', 'w') as summary:
        print('{} patients found.'.format(patient_id))
        summary.write('num_patients: {}\n'.format(patient_id))
        print('{} eye exams found.'.format(num_exams))
        summary.write('num_eye_exams: {}\n'.format(num_exams))
        print('Label histogram: {}'.format(dict(num_labels)))
        summary.write('label_histogram: {}\n'.format(dict(num_labels)))


if __name__ == "__main__":
    main()
