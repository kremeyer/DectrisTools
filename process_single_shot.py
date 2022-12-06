"""example file for processing a single-shot dataset
this script should be run with something like `mpirun -n XX python3 process_single_shot.py | tee processing.log`
"""
from pathlib import Path
from datetime import datetime
from mpi4py import MPI
import numpy as np
from DectrisTools.lib.processing import process_pump_probe

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

mask = np.load("mask.npy").astype(np.uint16)
rundir = "/data/run_0145"
filelist = sorted([str(p) for p in Path(rundir).rglob("*ps.h5")])
rois = {
    "bragg_1": (slice(172, 186), slice(126, 140)),
    "bragg_2": (slice(431, 444), slice(346, 360)),
    "interesting_diffuse_feature": (slice(404, 418), slice(359, 373)),
}

for i, file in enumerate(filelist):
    if i % size == rank:
        warns = process_pump_probe(file, mask=mask, rois=rois)
        print(
            f"{datetime.strftime(datetime.now(), '%Y/%m/%d %H:%M:%S')}: rank {rank:03d} processed {file} [{100 * i / len(filelist):.2f}%]"
        )
        for warn in warns:
            print(f"rank {rank:03d} encountered {type(warn).__name__} with message: {str(warn)}")
