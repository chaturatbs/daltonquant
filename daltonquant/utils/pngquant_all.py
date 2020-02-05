from argparse import ArgumentParser
import os
import sys

from ..compressors import PngQuant

def pngquant_all_images(inputdir, outputdir):
  pngquant = PngQuant()
  im_paths = [os.path.join(inputdir, f) for f in os.listdir(inputdir)]
  for im_path in im_paths:
    print("Applying pngquant to %s" % im_path)
    base_name = os.path.basename(im_path)
    outpath = os.path.join(outputdir, base_name)
    pngquant.compress(im_path, ncolors=256, outpath=outpath)

def main(args):
  if not os.path.exists(args.output_dir):
    print("Creating directory %s" % args.output_dir)
    os.mkdir(args.output_dir)
  if not os.path.exists(args.input_dir):
    raise Exception("Need directory for kodak images: %s" % args.input_dir)
  print("Quantizing all images in %s to %s" % (args.input_dir, args.output_dir))
  pngquant_all_images(args.input_dir, args.output_dir)

if __name__ == "__main__":
  parser = ArgumentParser(description="Apply pngquant 256 --speed 1 to all images in input folder")
  parser.add_argument('input_dir', type=str, help='Input directory with images (likely kodak folder)')
  parser.add_argument('output_dir', type=str, help='Output directory to store pngquantized images')
  args = parser.parse_args()
  main(args)