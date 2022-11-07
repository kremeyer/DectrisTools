"""module to process single shot dataset given the arguments `rundir` and `max_workers`
"""
from pathlib import Path
from concurrent.futures import as_completed
from tqdm import tqdm
from .lib.processing import SingleShotProcessor
import sys

if __name__ == "__main__":
    rundir = sys.argv[1]
    try:
        max_workers = sys.argv[2]
    except IndexError:
        max_workers = None
    filelist = [str(p) for p in Path(rundir).rglob("*s.h5")]
    with SingleShotProcessor(filelist, max_workers=max_workers) as ssp:
        ssp.start()
        for future in tqdm(as_completed(ssp.futures), total=len(filelist)):
            pass
