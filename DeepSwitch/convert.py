import numpy as np
import cv2
from glob import glob
import os
import sys

def convert(f, d):
  basename = os.path.splitext(f)[0]
  img = cv2.imread(f)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, dsize=(256, 192), interpolation=cv2.INTER_AREA)
  np.save(os.path.join(d, "{}.npy".format(basename)), img)

def process(d, c):
  files = glob(os.path.join(d, c, "*.png"))
  print("Number of Samples: {} {}".format(len(files), c))

  for f in files:
    convert(f, os.path.join(d, c))

if __name__ == "__main__":
  print("Converting dataset from PNG to NPY...")
  
  input_dir = sys.argv[1]
  classes = next(os.walk(input_dir))[1]
  print("Input Directory: {}".format(input_dir))
  print("Number of Classes: {}".format(len(classes)))

  for c in classes:
    process(input_dir, c)
  
