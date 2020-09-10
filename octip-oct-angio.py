"""
OCTIP script for generating 2-D angiograms from PLEXElite data.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'Gwenole Quellec (gwenole.quellec@inserm.fr)'
__copyright__ = 'Copyright (C) 2020 LaTIM'
__license__ = 'Proprietary'
__version__ = '1.1'


import cv2
import numpy as np
import octip
import os
import sys
from argparse import ArgumentParser


def main():
    """
    Generates 2-D angiograms from PLEXElite data.
    """

    # parsing the command line
    parser = ArgumentParser(
        description = 'Generates 2-D angiograms from PLEXElite data.')
    parser.add_argument('-i', '--input_dir', required = True,
                        help = 'directory containing PLEXElite DICOMs from one eye exam')
    parser.add_argument('-o', '--output_dir', default = '.',
                        help = 'directory containing 2-D angiograms')
    parser.add_argument('-l', '--rpe_layer', default = 'RPE_Fit_Layer',
                        help = 'RPE layer to use (RPE_Layer or RPE_Fit_Layer)')
    parser.add_argument('-f', '--multiplicative_factor', default = '1.', type = float,
                        help = 'angiogram intensity multiplicative factor')
    parser.add_argument('-r', '--ranges', default = [0., 1.], type = float, nargs = '+',
                        help = 'ranges of relative depths to use (pairs of values in [0; 1])')
    if len(sys.argv[1:]) == 0:
        parser.print_usage()
        parser.exit()
    args = parser.parse_args()

    # processing ranges
    assert len(args.ranges) % 2 == 0, 'ranges should contain an even number of cutoffs'
    for cutoff in args.ranges:
        assert 0. <= cutoff <= 1., 'ranges should contain values in [0; 1]'
    lbs, ubs = [], []
    for i, cutoff in enumerate(args.ranges):
        if i % 2 == 0:
            lbs.append(cutoff)
        else:
            ubs.append(cutoff)
    for lb, ub in zip(lbs, ubs):
        assert lb <= ub, 'lower bounds must be lower than upper bounds in ranges...'

    # function for unpacking one element in a list
    def single_element(input_list, name):
        assert len(input_list) == 1, 'no {} or more than one'.format(name)
        return input_list[0]

    # parsing the inputs
    parser = octip.PLEXEliteParser(args.input_dir)
    flow_volumes = single_element(parser.load_images(octip.PLEXEliteFileType.FLOW_CUBE),
                                  'flow volume')
    num_frames, num_rows, num_cols = flow_volumes.shape[0], flow_volumes.shape[1], \
        flow_volumes.shape[2]
    ilm_segmentation = single_element(parser.load_segmentations('ILM_Layer', num_frames),
                                      'ILM segmentation')
    rpe_segmentation = single_element(parser.load_segmentations(args.rpe_layer, num_frames),
                                      'RPE segmentation')

    # output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # computing angiograms
    cumulative_sum = np.cumsum(flow_volumes, axis = 1, dtype = float)
    delta = rpe_segmentation - ilm_segmentation
    idx = np.ix_(range(num_frames), range(num_rows), range(num_cols))
    for lb, ub in zip(lbs, ubs):

        # summing intensities in the range
        min_z = np.round(ilm_segmentation + lb * delta)
        max_z = np.round(ilm_segmentation + ub * delta)
        max_z = np.clip(max_z, min_z, num_rows).astype(np.int32)
        min_z = np.clip(min_z - 1, 0, num_rows).astype(np.int32)
        sum = cumulative_sum[idx[0], np.reshape(max_z, [num_frames, 1, num_cols]), idx[2]] \
            - cumulative_sum[idx[0], np.reshape(min_z, [num_frames, 1, num_cols]), idx[2]]
        sum = np.reshape(sum, [num_frames, num_cols])

        # saving the angiogram
        angiogram = np.clip(sum * args.multiplicative_factor / num_rows, 0., 255.).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, 'angiogram_{}_{}.png'.format(lb, ub)), angiogram)


if __name__ == "__main__":
    main()
