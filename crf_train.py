#!/usr/bin/env python

import argparse
from linearChainCRF import LinearChainCRF

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("datafile", help="data file for training input")
    # parser.add_argument("modelfile", help="the model file name. (output)")
    #
    # args = parser.parse_args()

    crf = LinearChainCRF()
    # crf_train.py data/chunking_small/small_train.data small_model.json
    # crf.train("data/chunking_small/smallest_train.data", "smallest_model.json")
    crf.train("data/chunking_small/smallest_train.data", "small_model.json")
    # crf.train("data/chunking_full/full_train.data", "full_model.json")
