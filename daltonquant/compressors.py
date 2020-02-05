# DaltonQuant uses multiple pre-quantizers
# and a generic incremental quantization implementation
# (where the color-merger decisions are based on scorers from scorers.py)
# This file contains all compressor implementations

import os
import shutil
import sys
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist, squareform

from specimen import utils as s_utils

from .utils import pil as p_utils
from .utils import compress as c_utils



class Compressor(object):
  """ Generic compressor, base class """
  def compress(self, impath):
    raise NotImplementedError('subclass implementations')

  def _make_path(self, impath, outdir=None, extra=None):
    """ Create a path for the output of a compressor, given the path to the original"""
    info = c_utils.get_path_info(impath)
    if outdir is None:
      outdir = info['dir']
    if extra is None:
      extra = ''
    if not isinstance(extra, str) and s_utils.is_iterable(extra):
      extra = '_'.join(extra)
    prefix = self.compression_method if len(extra) == 0 else self.compression_method + '_' + extra
    newname = "%s_%s%s" % (prefix, info['name'], info['ext'])
    newpath = os.path.join(outdir, newname)
    return newpath


class PngQuant(Compressor):
  """ Wrapper around pngquant command line tool """
  def __init__(self):
    self.compression_method = 'pngquant'

  def compress(self, impath, ncolors=256, outpath=None, outdir=None, ret_im=False):
    if outpath is None:
      outpath = self._make_path(impath, outdir=outdir, extra='%dncolors' % ncolors)
    # slowest as we don't care about performance right now, just quality
    opts = [str(ncolors), '--force', '--speed', '1', '--output', outpath]
    c_utils.pngquant(impath, opts)
    if ret_im:
      return Image.open(outpath)
    else:
      return outpath


class TinyPng(Compressor):
  """ Wrapper around tiny png online API """
  def __init__(self, tinypng_key, store_folder):
    # may want to have multiple instances associated with diff api keys
    self.tinify = __import__('tinify')
    self.compression_method = 'tinypng'
    if not os.path.exists(store_folder):
      raise ValueError('Store folder for tinypng must exist or be created')
    self.store_folder = store_folder
    # we can pass the key in a file, if so, read it
    # key is a convoluted string, so unlikely to conflict with existing file
    if os.path.exists(tinypng_key):
      with open(tinypng_key, 'r') as f:
        tinypng_key = f.read().strip()
    self.tinify.key = tinypng_key

  def compress(self, impath, outpath=None, outdir=None, ret_im=False):
    info = c_utils.get_path_info(impath)
    if outpath is None:
      outpath = self._make_path(impath, outdir=outdir)
    # check if we already have a cached version, and can avoid hitting request limit (~500 per month)
    cached_path = os.path.join(self.store_folder, info['file'])
    # if it wasn't in our cache, let's hit API for request and save to cache
    if not os.path.exists(cached_path):
      compressed = self.tinify.from_file(impath)
      compressed.to_file(cached_path)
    # copy file from cached location to desired
    shutil.copy(cached_path, outpath)
    if ret_im:
      return Image.open(outpath)
    else:
      return outpath


class UpperBoundQuant(Compressor):
  """ Compress image merging colors based on number of colors to keep and pixel frequency """
  def __init__(self):
    self.compression_method = 'upper-bound'

  def compress(self, impath, ncolors=256, outpath=None, outdir=None, ret_im=None):
    if ncolors > 256:
      raise ValueError('Upper bound quantization must have <= 256 colors')
    im = Image.open(impath)
    if im.mode != 'RGB':
      im = im.convert('RGB')
    if outpath is None:
      outpath = self._make_path(impath, outdir=outdir, extra='%dncolors' % ncolors)
    compressed = self._compress_rgb_image(im, ncolors)
    compressed.save(outpath)
    if ret_im:
      return compressed
    else:
      return outpath

  def _reduced_colorspace(self, im, target_ncolors):
    color_cts = p_utils.count_colors(im)
    color_cts = color_cts.sort_values('ct', ascending=False)
    ncolors = color_cts.shape[0]
    remove = color_cts.iloc[:(ncolors - target_ncolors + 1)]
    remove = remove[list('rgb')].apply(tuple, axis=1).values
    replacement = tuple(color_cts.iloc[0][list('rgb')])
    return {c:replacement for c in remove}

  def _compress_rgb_image(self, im, ncolors):
    """ Works on RGB-format (reduces colors based on pixel frequency) """
    assert(im.mode == 'RGB')
    orig_palette = p_utils.count_colors(im)[list('rgb')]
    orig_palette = orig_palette.apply(tuple, axis=1).values
    # all colors originally map to themselves
    color_map = {c: c for c in orig_palette}
    # and we update this with our reduced color space
    remapping = self._reduced_colorspace(im, ncolors)
    color_map.update(remapping)
    # create new color palette based on values that remain
    new_palette = np.array(list(set(color_map.values())))
    # map each color to index in map, needed for palette images
    new_palette_ix_map = dict(list(zip(list(map(tuple, new_palette)), list(range(new_palette.shape[0])))))
    get_new_index = lambda orig_color: new_palette_ix_map[color_map[orig_color]]
    # add as many necessary zeros to create complete 768 entry palette
    zeros = np.tile([0, 0, 0], (256 - ncolors, 1))
    new_palette_ext = np.vstack([new_palette, zeros])
    new_palette_ext = new_palette_ext.flatten(order='C').tolist()
    # TODO: fix this! this is a hacky version to get PIL to realize we've reduced the palette!
    compressed_im = im.convert('P', palette=Image.ADAPTIVE, colors=ncolors)
    compressed_im.putpalette(new_palette_ext)
    # change up the mappings into the palette
    orig_pixels = im.load()
    compressed_pixels = compressed_im.load()
    nrows, ncols = im.size
    # TODO: faster
    for r in range(nrows):
      for c in range(ncols):
        compressed_pixels[r, c] = get_new_index(orig_pixels[r, c])

    return compressed_im

class MedianCutQuant(Compressor):
  """ Median cut quantizer """
  def __init__(self):
    self.compression_method = "mediancut"

  def compress(self, impath, ncolors=256, outpath=None, outdir=None, ret_im=None):
    info = c_utils.get_path_info(impath)
    im = Image.open(impath)
    if im.mode is not 'RGB':
      im = p_utils.to_rgb(im)
    if outpath is None:
      outpath = self._make_path(impath, outdir=outdir)
    compressed = self._compress_rgb_image(im, ncolors)
    compressed.save(outpath)
    if ret_im:
      return compressed
    else:
      return outpath

  def _reduced_colorspace(self, im, target_ncolors):
    color_cts = p_utils.count_colors(im)
    color_cts = color_cts.sort_values('ct', ascending=False)
    return self._median_cut(color_cts, target_ncolors - 1)

  def _median_cut(self, colors, target_ncolors):
    # we ran out of colors, just return empty dict to update in recursive step
    if colors.shape[0] == 0:
      return {}
    # for this current slice, replace all colors with the mean in each dimension (rgb)
    if target_ncolors <= 1:
      # reduce and apply floor by converting to int
      replacement = tuple(colors[list('rgb')].mean(axis=0).astype(int))
      # map colors to mean value
      old = colors[list('rgb')].apply(tuple, axis=1)
      return {c: replacement for c in old}
    else:
      # find dimension of largest range
      mins = colors[list('rgb')].min(axis=0)
      maxs = colors[list('rgb')].max(axis=0)
      ranges = np.abs(maxs - mins)
      # returns r, g, b, depending on dimension with largest range
      argmax_dim = np.argmax(ranges)
      # sort colors along this dim
      sorted_colors = colors.sort_values(argmax_dim, ascending=True)
      split_pt = sorted_colors.shape[0] / 2
      h1 = sorted_colors.iloc[:split_pt]
      h2 = sorted_colors.iloc[split_pt:]
      h1_ncolors = target_ncolors / 2
      h1_replacements = self._median_cut(h1, h1_ncolors)
      h2_replacements = self._median_cut(h2, target_ncolors - h1_ncolors)
      h1_replacements.update(h2_replacements)
      return h1_replacements

  def _compress_rgb_image(self, im, ncolors):
    """ Works on RGB-format (reduces colors based on pixel frequency) """
    assert(im.mode == 'RGB')
    orig_palette = p_utils.count_colors(im)[list('rgb')]
    orig_palette = orig_palette.apply(tuple, axis=1).values
    # all colors originally map to themselves
    color_map = {c: c for c in orig_palette}
    # and we update this with our reduced color space
    remapping = self._reduced_colorspace(im, ncolors)
    color_map.update(remapping)
    # create new color palette based on values that remain
    new_palette = np.array(list(set(color_map.values())))
    # map each color to index in map, needed for palette images
    new_palette_ix_map = dict(list(zip(list(map(tuple, new_palette)), list(range(new_palette.shape[0])))))
    get_new_index = lambda orig_color: new_palette_ix_map[color_map[orig_color]]
    # add as many necessary zeros to create complete 768 entry palette
    zeros = np.tile([0, 0, 0], (256 - ncolors, 1))
    new_palette_ext = np.vstack([new_palette, zeros])
    new_palette_ext = new_palette_ext.flatten(order='C').tolist()
    # TODO: fix this! this is a hacky version to get PIL to realize we've reduced the palette!
    compressed_im = im.convert('P', palette=Image.ADAPTIVE, colors=ncolors)
    compressed_im.putpalette(new_palette_ext)
    # change up the mappings into the palette
    orig_pixels = im.load()
    compressed_pixels = compressed_im.load()
    nrows, ncols = im.size
    # TODO: faster
    for r in range(nrows):
      for c in range(ncols):
        compressed_pixels[r, c] = get_new_index(orig_pixels[r, c])

    return compressed_im

class SpecimenQuant(Compressor):
  """ Compress image using specimen-based information """
  def __init__(self, scorer, quantizer=None):
    self.quantizer = PngQuant() if quantizer is None else quantizer
    self.compression_method = 'specimen_%s_%s' % (scorer.scorer_method, quantizer.compression_method)
    self.scorer = scorer

  def compress(self, impath, target_ncolors, alpha, outpath=None, outdir=None, ret_im=False, ret_inv=False):
    im = Image.open(impath)
    if im.mode != 'P':
      # if not already paletted, quantize with pngquant
      temp = NamedTemporaryFile(suffix='.png')
      im = self.quantizer.compress(impath, outpath=temp.name, ret_im=True)
      temp.close()
    # run specimen stuff here
    if outpath is None:
      # details about compression used
      extra = ['userid', str(self.scorer.userid), 'ncolors', '%d' % target_ncolors]
      outpath = self._make_path(impath, outdir=outdir, extra=extra)
    # compress the already paletted image further
    compressed, reverse_mapping = self._compress_palette_image(im, target_ncolors, alpha)
    compressed.save(outpath)
    if ret_im:
      out_val = compressed
    else:
      out_val = outpath
    if ret_inv:
      return out_val, reverse_mapping
    else:
      return out_val

  def _compress_palette_image(self, im, target_ncolors, alpha):
    """ Works on P-format (reduces 256-bit color palette further based on scorer ranking)"""
    assert(im.mode == 'P')
    orig_palette = np.array(im.getpalette()).reshape(-1, 3)
    # all colors originally map to themselves
    color_map = {c: c for c in map(tuple, orig_palette)}
    # and we update this with our reduced color space
    remapping = self._reduced_colorspace(p_utils.to_rgb(im), target_ncolors, alpha)
    color_map.update(remapping)
    return self._apply_new_color_map(im, orig_palette, color_map)

  def _apply_new_color_map(self, im, orig_palette, color_map):
    # create new color palette based on values that remain
    new_colors = set(color_map.values())
    new_palette = np.array(list(new_colors))
    # map each color to index in map, needed for palette images
    new_palette_ix_map = dict(list(zip(list(map(tuple, new_palette)), list(range(new_palette.shape[0])))))
    get_new_palette_index = lambda col: new_palette_ix_map[color_map[col]]
    # figure out how many colors we have
    new_ncolors = len(new_colors)
    # add as many necessary zeros to create complete 768 entry palette
    zeros = np.tile([0, 0, 0], (256 - new_ncolors, 1))
    new_palette_ext = np.vstack([new_palette, zeros])
    new_palette_ext = new_palette_ext.flatten(order='C').tolist()
    # TODO: fix this! this is a hacky version to get PIL to realize we've reduced the palette!
    compressed_im = im.convert('P', palette=Image.ADAPTIVE, colors=new_ncolors)
    # we put in the palette we actually wanted
    compressed_im.putpalette(new_palette_ext)
    # change up the mappings into the palette
    orig_pixels = im.load()
    compressed_pixels = compressed_im.load()
    nrows, ncols = im.size
    # for pixels that are recolored, we produce an inverse mapping
    changed_pixel_mapping = {}
    # TODO: faster
    for r in range(nrows):
      for c in range(ncols):
        # retrieve color from original palette
        orig_color = tuple(orig_palette[orig_pixels[r, c]])
        if not orig_color in new_colors:
          changed_pixel_mapping[(r, c)] = orig_color
        # use updated index mapping since colors move around in the palette
        compressed_pixels[r, c] = get_new_palette_index(orig_color)
    return compressed_im, changed_pixel_mapping

  def inverse(self, im, pixel_to_color_map):
    assert(im.mode == 'P')
    # all colors in inv_map need to be added, append at the end of the current palette
    new_im = im.copy()
    new_palette = np.array(new_im.getpalette()).reshape(-1, 3)
    missing_colors = set(pixel_to_color_map.values())
    palette_index_map = {}
    offset = 256 - len(missing_colors)
    assert(offset >= 0)
    for i, col in enumerate(missing_colors):
      ix = offset + i
      new_palette[ix, :] = col
      palette_index_map[col] = ix
    # flatten out palette before adding to image
    new_palette = new_palette.flatten(order='C').tolist()
    new_pixels = new_im.load()
    new_im.putpalette(new_palette)
    for (r, c), col in list(pixel_to_color_map.items()):
      new_pixels[r, c] = palette_index_map[col]
    return new_im

  def _transitive_closure(self, m):
    """ Compute transitive closure of a dictionary, such that if X -> Y, Y -> Z => X -> Z """
    m = m.copy()
    changed = True
    while changed:
      changed = False
      for k, v in m.items():
        if v in m:
          m[k] = m[v]
          changed = True
    return m

  def _reduced_colorspace(self, im, target_ncolors, alpha):
    color_cts = p_utils.count_colors(im)
    if color_cts.shape[0] > 256 and target_ncolors < 256:
      print("Error: Currently only works on pre-processed images with reduced color spaces, apply pngquant first")
      sys.exit(1)
    candidates = self.scorer.sorted_candidates(color_cts, alpha)
    require = 256 - target_ncolors
    merge = {}
    # go down the list and for each color option 1, take only
    # the "highest" ranked color option 2
    for c1, c2, _ in candidates:
      if tuple(c1) in merge:
        continue
      else:
        require -= 1
        merge[tuple(c1)] = tuple(c2)
      if require == 0:
        break
    # closure to recolor anything transitively if necessary
    return self._transitive_closure(merge)
