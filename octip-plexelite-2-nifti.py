"""
OCTIP script for converting PLEXElite data in compressed NifTI format.
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
import sys
from argparse import ArgumentParser
from scipy import ndimage


def main():
    """
    Converts PLEXElite data in compressed NifTI format.
    """

    # parsing the command line
    parser = ArgumentParser(
        description = 'Converts PLEXElite data in compressed NifTI format.')
    parser.add_argument('-i', '--input_dirs', required = True, nargs = '+',
                        help = 'space-delimited list of input directories')
    parser.add_argument('-v', '--volume_dir', default = 'volumes',
                        help = 'directory containing volumes in compressed NifTI format')
    parser.add_argument('--image_dir', default = None,
                        help = 'directory containing selected and resized 2-D images '
                               '(can be safely removed at the end of this script)')
    parser.add_argument('-d', '--depth_selection', default = 1., type = float,
                        help = 'Percentage of the depth that is selected')
    parser.add_argument('-D', '--output_depth', default = None, type = int,
                        help = 'Output depth (no resizing in depth by default)')
    parser.add_argument('-l', '--rpe_layer', default = 'None',
                        help = 'If RPE layer (RPE_Layer or RPE_Fit_Layer) provided, zero below')
    if len(sys.argv[1:]) == 0:
        parser.print_usage()
        parser.exit()
    args = parser.parse_args()

    # output volume directory
    if not os.path.exists(args.volume_dir):
        os.makedirs(args.volume_dir)

    # function for unpacking one element in a list
    def single_element(input_list, name):
        assert len(input_list) > 0, 'no {}'.format(name)
        assert len(input_list) < 2, 'more than one {}'.format(name)
        return input_list[0]

    # loop on studies
    with open('laterality.csv', 'w') as f:
        f.write('url,laterality\n')
    total_num_acquisitions = 0
    for main_dir in args.input_dirs:
        for exam in glob.glob(os.path.join(main_dir, '*')):
            subdir = os.path.join(exam, 'DATAFILES')
            if os.path.exists(subdir):
                print('Processing exam \'{}\'...'.format(exam))
                num_valid_acquisitions = 0
                for acquisition_dir in glob.glob(os.path.join(subdir, '*')):
                    acquisition = os.path.split(acquisition_dir)[1]
                    print('  Processing acquisition \'{}\'...'.format(acquisition))
                    volume_name = exam + '_' + acquisition

                    # parsing the inputs
                    try:
                        parser = octip.PLEXEliteParser(acquisition_dir)
                        flow_volume, laterality = single_element(
                            parser.load_images(octip.PLEXEliteFileType.FLOW_CUBE, True),
                            'flow volume')
                        structure_volume = single_element(
                            parser.load_images(octip.PLEXEliteFileType.STRUCTURE_CUBE),
                            'structure volume')
                        assert flow_volume.shape == structure_volume.shape, \
                            'flow and structure volumes have different shapes!'
                        num_frames, num_rows, num_cols = flow_volume.shape[0], \
                            flow_volume.shape[1], flow_volume.shape[2]
                        ilm_segmentation = single_element(
                            parser.load_segmentations('ILM_Layer', num_frames), 'ILM segmentation')
                        rpe_segmentation = single_element(
                            parser.load_segmentations(args.rpe_layer, num_frames),
                            'RPE segmentation') if args.rpe_layer is not None else None
                        print('Laterality: "{}"'.format(laterality))
                    except Exception as err:
                        #print('Error acquisition \'{}\'...'.format(acquisition, err))
                        continue

                    # voxel coordinates
                    x, y, z = np.ix_(range(num_frames), range(num_rows), range(num_cols))

                    # masking out the choroid if RPE segmentation is specified
                    if rpe_segmentation is not None:
                        mask = np.zeros_like(flow_volume, dtype = flow_volume.dtype)
                        mask[np.where(y <= rpe_segmentation[:, np.newaxis, :])] = 1
                        flow_volume = flow_volume * mask
                        structure_volume = structure_volume * mask

                    # shifting A-scan indices
                    shift = num_rows - ilm_segmentation
                    y = y - shift[:, np.newaxis, :]
                    depth = int(num_rows * args.depth_selection) \
                        if args.depth_selection < 1. else None

                    # sub-volume extraction function
                    def extract(volume):
                        volume = volume[x, y, z]
                        if depth:
                            volume = volume[:, :depth, :]
                        return volume[:, :, :, np.newaxis]

                    # extracting and concatenating the sub-volumes
                    flow_volume = extract(flow_volume)
                    structure_volume = extract(structure_volume)
                    sub_volumes = np.concatenate([flow_volume, structure_volume], axis = 3)

                    # resizing in depth if requested
                    if args.output_depth is not None:
                        eps = 1e-7
                        scale = (args.output_depth + eps) / sub_volumes.shape[1]
                        sub_volumes = ndimage.zoom(sub_volumes, zoom = [1., scale, 1., 1.],
                                                   order = 1)
                        if sub_volumes.shape[1] > args.output_depth:
                            trim = (sub_volumes.shape[1] - args.output_depth) // 2
                            sub_volumes = sub_volumes[:, trim:trim + args.output_depth, :, :]

                    # saving as images
                    if args.image_dir:
                        output_dir = os.path.join(args.image_dir, volume_name)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        for i in range(num_frames):
                            image = np.concatenate([sub_volumes[i, :, :, 0],
                                                    sub_volumes[i, :, :, 1]], axis = 1)
                            cv2.imwrite(os.path.join(output_dir, 'frame{}.png'.format(i)),
                                        image[:, :])

                    # saving the volume
                    img = nib.Nifti1Image(sub_volumes, np.eye(4))
                    nib.save(img, os.path.join(args.volume_dir, volume_name + '.nii.gz'))

                    # saving laterality
                    with open('laterality.csv', 'a') as f:
                        f.write('{},{}\n'.format(os.path.basename(volume_name + '.nii.gz'), laterality))

                    num_valid_acquisitions += 1
                total_num_acquisitions += num_valid_acquisitions
                print('  {} acquisition(s) found (total: {}).'.format(num_valid_acquisitions,
                                                                      total_num_acquisitions))


if __name__ == "__main__":
    main()
