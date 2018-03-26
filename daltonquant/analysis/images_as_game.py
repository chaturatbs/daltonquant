import argparse
import os

import colorsys
import numpy as np
import pandas as pd
from PIL import Image

import specimen.utils as s_utils

from ..utils import pil as p_utils


def image_to_pixels(im):
  im = p_utils.to_rgb(im)
  h, w = im.size
  pixels = np.array(im)
  return pixels.reshape(h * w, 3)
  
def images_to_specimen(target_path, specimen_path):
  print "Comparing %s[target] vs %s[specimen]" % (target_path, specimen_path)
  target = Image.open(target_path)
  specimen = Image.open(specimen_path)
  pixels_t = image_to_pixels(target)
  pixels_s = image_to_pixels(specimen)
  correct = (pixels_t == pixels_s).all(axis=1)
  rgb_to_hue = lambda x: colorsys.rgb_to_hsv(*x)[0]
  # divide pixels (which are in RGB) by 255 to be within 0 and 1.0
  hues_t = np.apply_along_axis(rgb_to_hue, 1, pixels_t / 255.)
  hue_bucket = s_utils.munsell_buckets(hues_t, labels=True)[1]
  df = pd.DataFrame({'hue': hue_bucket, 'correct': correct})
  return df.groupby('hue')['correct'].agg({'ct': len, 'correct': np.sum}).reset_index()
  
def hue_accuracies(targets, specimens):
  dfs = [images_to_specimen(t, s) for t, s in zip(targets, specimens)]
  results = pd.concat(dfs, axis=0).groupby('hue').sum().reset_index()
  results['acc'] = results['correct'] / results['ct']
  return results


if __name__ == '__main__':
  description = """
  Treat images, pixel-wise as a specimen game. Same pixel color == correct selection. Calculate
  hue-level accuracies
  """
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('targets', type=str, help='csv sep list of paths to original images')
  parser.add_argument('specimens', type=str, help='csv sep list of paths to specimen images')
  parser.add_argument('-o', '--out', type=str, help='Output path for csv', default='simulated.csv')
  args = parser.parse_args()
  targets = map(lambda x: x.strip(), ','.split(args.targets))
  specimens = map(lambda x: x.strip(), ','.split(args.specimens))
  result = analyze(targets, specimens)
  result.to_csv(args.out, index=False)

# TODO: for 0.05 to 0.2, for each user, run this script 
# TODO: create plot: X axis hue bucket, bar plots, grouped by alpha

