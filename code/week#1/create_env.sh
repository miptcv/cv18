#!/usr/bin/env bash
conda create -n miptcv python=2.7 ipython opencv matplotlib

conda install opencv

source activate miptcv
source deactivate
