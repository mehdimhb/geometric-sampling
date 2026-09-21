#!/bin/sh
set -eu
exec env PYTHONPATH=/home/bardia/projects/graphical-sampling/prescribed_sip/final_equal_fip/configs/code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gfs-final-mpl /home/bardia/projects/graphical-sampling/.venv/bin/python -u -m prescribed_sip.final_study --output /home/bardia/projects/graphical-sampling/prescribed_sip/final_equal_fip --fip-type equal
