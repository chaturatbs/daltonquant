# transform an image into the palette of a given user
from argparse import ArgumentParser
import os

import colorsys
from PIL import Image
import numpy as np
import pandas as pd

import specimen.utils as s_utils

from ..scorers import LinearTransform, NonLinearTransform


def see_image(quantized_image, scorer):
  transformed_image = quantized_image.copy()
  rgb_palette = np.array(transformed_image.getpalette()).reshape((-1, 3))
  #  convert
  lab_palette = s_utils.mat_rgb_to_lab(rgb_palette / 255.0)
  transformed_lab_palette = scorer._transform(lab_palette)
  # convert back to rgb
  transformed_rgb_palette = s_utils.mat_lab_to_rgb(transformed_lab_palette) * 255.0
  transformed_rgb_palette = np.round(transformed_rgb_palette).astype(int)
  # fix bounds
  is_neg = np.where(transformed_rgb_palette < 0)
  transformed_rgb_palette[is_neg] = 0
  is_gt_255 = np.where(transformed_rgb_palette > 255.)
  transformed_rgb_palette[is_gt_255] = 255
  transformed_image.putpalette(transformed_rgb_palette.flatten(order='C'))
  return transformed_image, rgb_palette, transformed_rgb_palette

def transform(userid, database_path, transform_type, im_path, output_dir):
  im = Image.open(im_path)
  if transform_type == 'linear':
    transform_constructor = LinearTransform
  elif transform_type == 'non-linear':
    transform_constructor = NonLinearTransform
  else:
    raise Exception("Unknown transform type, must be one of [linear, non-linear]")
  
  transformer = transform_constructor(userid, database_path=database_path)
  transformer.fit()
  transformed_im = see_image(im, transformer)[0]
  transformed_name = '%s-%s-transformed.png' % (os.path.basename(im_path).split('.')[0], transform_type)
  transformed_path = os.path.join(output_dir, transformed_name)
  transformed_im.save(transformed_path)
  #
  # nlt_impath = "ext_compression_results/specimen_nonlinear transform (nn)_pngquant_userid_50165_ncolors_204_kodim07.png"
  # nlt_im = Image.open(nlt_impath)
  # nlt = NonLinearTransform(50165, database_path=database_path)
  # nlt.fit()
  # nlt_transformed = see_image(nlt_im, nlt)[0]
  # nlt_transformed.save('analysis_results/kodim07_nlt_transformed.png')
  #
  # lt_impath = "ext_compression_results/specimen_linear transform (least-squares)_pngquant_userid_50165_ncolors_204_kodim07.png"
  # lt_im = Image.open(lt_impath)
  # lt = LinearTransform(50165)
  # lt.fit()
  # lt_transformed = see_image(lt_im, lt)[0]
  # lt_transformed.save('analysis_results/kodim07_lt_transformed.png')


def main(args):
  args.input_paths = args.input_paths.split(',')
  args.transform_types = args.transform_types.split(',')
  for im_path, transform_type in zip(args.input_paths, args.transform_types):
    transform(args.userid, args.database_path, transform_type, im_path, args.output_dir)
  
if __name__ == "__main__":
  parser = ArgumentParser(description='Apply transform to image and show how it looks')
  parser.add_argument('database_path', type=str, help='Path to specimen database file')
  parser.add_argument('userid', type=str, help='User id to visualize')
  parser.add_argument('input_paths', type=str, help='csv list of paths to input images to transform')
  parser.add_argument('transform_types', type=str, help='csv list of [linear, non-linear] transforms to apply')
  parser.add_argument('output_dir', type=str, help='Directory to output to')
  args = parser.parse_args()
  try:
    main(args)
  except Exception as err:
    import pdb
    pdb.post_mortem()
  
  