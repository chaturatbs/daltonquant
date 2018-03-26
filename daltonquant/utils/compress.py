import os
from subprocess import Popen, PIPE

# file system related utilities used commonly for compression
# routines
def is_iterable(x):
  try:
    iter(x)
    return True
  except TypeError:
    return False

def get_path_info(file_path):
  """ return relevant path info as dict (base directory, base file, file name w/o ext, ext)"""
  base_dir, base_file = os.path.split(file_path)
  nm, ext = os.path.splitext(base_file)
  return {'dir': base_dir, 'file': base_file, 'name': nm, 'ext': ext}
  
def file_size(file_path):
  """ file size as measured by stat -f %%z """
  # file sizes in bytes
  #p = Popen(['stat', '-f', '%z', file_path], stdout=PIPE)
  with open(file_path, 'r') as f:
    p = Popen(['wc', '-c'], stdin=f, stdout=PIPE)
  stdout, _ = p.communicate()
  return int(stdout.strip())
  
def pngquant(file_path, opts=None):
  """ run pngquant with defaults """
  if opts is None:
    p = Popen(['pngquant', file_path], stdout=PIPE)
  else:
    if not is_iterable(opts):
      raise ValueError('options must be iterable')
    opts = list(opts)
    p = Popen(['pngquant'] + opts + [file_path], stdout=PIPE)
  stdout, _ = p.communicate()
  return stdout

