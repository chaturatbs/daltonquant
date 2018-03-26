# Evaluate different compression algorithms and
# different settings

import argparse
import os
from tempfile import NamedTemporaryFile
import re

import pandas as pd
from PIL import Image

from ..compressors import PngQuant, TinyPng, UpperBoundQuant, MedianCutQuant, SpecimenQuant
from ..scorers import GlobalKDE, HueKDE, LinearTransform, NonLinearTransform
from ..utils import pil as p_utils
from ..utils import compress as c_utils

# some globals for tiny png config

DEBUG = False

def print_debug(msg):
  if DEBUG:
    print msg

def evaluate_image(origpath, impath):
  """ Evaluate an image """
  fs = c_utils.file_size(impath)
  rgb_im = p_utils.to_rgb(Image.open(impath))
  ncolors = p_utils.count_colors(rgb_im).shape[0]
  return {'fs': fs, 'ncolors': ncolors}
  

def evaluate(impath, scorer, quantizer, target_ncolors, outdir, alpha, tiny_png_key, tiny_png_cache_folder):
  path_info = c_utils.get_path_info(impath)
  # compressors
  pngquant = PngQuant()
  tinypng = TinyPng(tiny_png_key, tiny_png_cache_folder)
  medquant = MedianCutQuant()
  specimen = SpecimenQuant(scorer, quantizer=quantizer)
  print_debug("Evaluating userid:%s, target ncolors:%d on %s using %s and alpha %f" % (scorer.userid, target_ncolors, impath, specimen.compression_method, alpha))
  
  # paths to compressed images
  print_debug("specimen compression")
  specimen_path = specimen.compress(impath, target_ncolors, alpha, outdir=outdir, ret_im=False)
  print_debug("pngquant compression")
  pngquant_path = pngquant.compress(impath, ncolors=target_ncolors, outdir=outdir, ret_im=False)
  print_debug("tinypng compression")
  tinypng_path = tinypng.compress(impath, outdir=outdir, ret_im=False)
  print_debug("median cut compression")
  medquant_path = medquant.compress(impath, ncolors=target_ncolors, outdir=outdir, ret_im=False)

  # evaluate results by different metrics
  result_metrics = {}
  result_metrics['method'] = specimen.compression_method
  result_metrics['userid'] = scorer.userid
  result_metrics['image'] = impath
  result_metrics['target_ncolors'] = target_ncolors

  methods = ['orig', 'specimen', 'pngquant', 'tinypng', 'medquant']
  paths = [impath, specimen_path, pngquant_path, tinypng_path, medquant_path]
  methods_and_path = zip(methods, paths)
  
  for method, compressed_path in methods_and_path:
    metrics = evaluate_image(impath, compressed_path)
    metrics = {('%s_%s' % (method, k)): v for k, v in metrics.iteritems()}
    result_metrics.update(metrics)
  result_df = pd.DataFrame([result_metrics])
  # sort order of columns
  columns = ['method', 'userid', 'image', 'target_ncolors']
  # file sizes
  metrics = ['fs']
  columns += ['%s_%s' % (method, metric) for metric in metrics for method in methods]
  return result_df[columns]

def str_results(results):
  results = results.iloc[0]
  results.name = None
  return str(results)

def main(args):
  results = []
  # quantizers that specimen builds upon
  quantizers =[PngQuant(), TinyPng(args.tiny_png_key, args.tiny_png_cache_folder), MedianCutQuant()]
  scorer_constructors = [LinearTransform, NonLinearTransform]
  # count number of runs to provide some estimate
  nruns = len(args.userids) * len(scorer_constructors) * len(quantizers) * len(args.input_images) * len(args.target_ncolors)
  done = 0
  print "Running a total of %d evaluations" % nruns
  for user in args.userids:
    # construct distinct scorers for a given user
    user_scorers = [c(user, database_path=args.database_path) for c in scorer_constructors]
    for s in user_scorers:
      s.fit()
    for quant in quantizers:
      for scorer in user_scorers:
        for image in args.input_images:
          for target in args.target_ncolors:
            results.append(evaluate(image, scorer, quant, target, args.outdir, args.alpha, args.tiny_png_key, args.tiny_png_cache_folder))
            done += 1
            print "Done with %d/%d" % (done, nruns)
            

  results = pd.concat(results, axis=0)
  results_csv_file = os.path.join(args.outdir, 'results.csv')
  results.to_csv(results_csv_file, index=False)
  if args.stdout:
      for res in results:
        print str_results(res) + '\n'

def sarg_to_list(s):
  elems = s.split(',') if ',' in s else s.split()  
  elems = map(lambda x: x.strip(), elems)
  return [e for e in elems if len(e) > 0]


if __name__ == "__main__":
  description = """
  Evaluation script for incremental color quantization using Specimen data.
  Each iteration ((image, userid, target_ncolors)) is run with 4 distinct scorers:
  global and hue-specific density-based scorers, a linear and non-linear distance-based scorers.
  Each uses an initial quantization pass with 3 types of quantizers
  (pngquant, tinypng, median cut). Note that the first two are complete compressors, with more
  techniques than just color quantization.
  
  Stores statistics to results.csv in outputdir.
  """
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('database_path', type=str, help='Path to Specimen database file')
  parser.add_argument('input_images', type=str, help='csv or whitespace sep list of paths to original images')
  parser.add_argument('target_ncolors', type=str, help='csv or whitespace sep list of target number of colors for compression')
  parser.add_argument('tiny_png_key', type=str, help='A TinyPNG key string (see README if you do not have access to TinyPNG)')
  parser.add_argument('tiny_png_cache_folder', type=str, help='A folder to store TinyPNG compressions to avoid repeat queries')
  parser.add_argument('-o', '--outdir', type=str, help='Output directory for created images', default='.')
  parser.add_argument('-u', '--userids', type=str, help='csv or whitespace sep list of unique userids (uniqueId) from specimen data', default='')
  parser.add_argument('-d', '--debug', action='store_true', help="Print out some debugging info (dev)")
  parser.add_argument('-s', '--stdout', action='store_true', help="Print out results to stdout at end of run")
  parser.add_argument('-a', '--alpha', type=float, help="Multi-objective weight for specimen compression", default=0.5)
  args = parser.parse_args()
  DEBUG = args.debug

  args.input_images = sarg_to_list(args.input_images)
  args.target_ncolors = map(int, sarg_to_list(args.target_ncolors))
  args.userids = args.userids.split(',')
    
  
  print args
  try:
    main(args)
  except:
    import pdb
    pdb.post_mortem()
