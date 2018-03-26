import numpy as np
import pandas as pd
from PIL import Image


# PIL related
def to_rgb(im):
  """convert image to rgb"""
  if im.mode == 'RGB':
    return im
  else:
    return im.convert('RGB')
       
def count_colors(im):
  """ Table of color frequency (# of pixels) """
  if im.mode != 'RGB':
    raise ValueError('Counting colors with array approach only defined on RGB')
  nrow, ncol = im.size
  rgb = list('rgb')
  df = pd.DataFrame(np.array(im).reshape((nrow * ncol, 3)), columns=rgb)
  cts = df.groupby(rgb)[['r']].count().rename(columns = {'r' :'ct'}).reset_index()
  cts = cts.sort_values('ct', ascending=False).reset_index(drop=True)
  return cts
  
