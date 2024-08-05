import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None, required=True)
parser.add_argument('--output_dir', type=str, default=None, required=True)
parser.add_argument('--images_dir', type=str, default=None, required=True)
opt = parser.parse_args()

for pic in os.listdir(opt.images_dir):
  outname = "dp_"+pic+".npy"
  print(opt.input)
  os.system(f"cp {opt.input} {os.path.join(opt.output_dir,outname)}")