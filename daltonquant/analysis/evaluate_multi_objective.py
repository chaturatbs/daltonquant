from argparse import ArgumentParser
import os
import tempfile
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from ..compressors import PngQuant, SpecimenQuant
from ..scorers import NonLinearTransform, HueKDE
from .evaluate import evaluate_image


def evaluate_multi(database_path, user, target_ncolors, impaths, outdir):
  pngquant = PngQuant()
  scorer = NonLinearTransform(user, database_path=database_path)
  scorer.fit()
  specimen = SpecimenQuant(scorer, quantizer=pngquant)
  tempdir = tempfile.mkdtemp()

  alphas = np.arange(0.0, 1.1, 0.1)
  acc = []
  for impath in impaths:
    for alpha in alphas:
      print("%s alpha = %f" % (impath, alpha))
      specimen_impath = specimen.compress(impath, target_ncolors, alpha, outdir=tempdir)
      res = evaluate_image(impath, specimen_impath)
      res['alpha'] = alpha
      res['image'] = impath
      acc.append(res)

  df = pd.DataFrame(acc)
  df.to_csv(os.path.join(outdir, 'multi_objective_eval.csv'))
  size_at_1 = df[df['alpha'] == 1.0][['image', 'fs']].rename(columns={'fs': 'denom'})
  df = pd.merge(df, size_at_1, on=['image'])
  df['fs_decrease'] = df['fs'] / df['denom'] - 1

  df_summary = df.groupby('alpha')[['fs_decrease']].mean().reset_index()
  alpha_plot = sns.pointplot(data=df_summary, x='alpha', y='fs_decrease')
  alpha_plot.set_xlabel(r'$\alpha$ (Multi-Objective Weight)', size=15)
  alpha_plot.set_ylabel(r'Average File Size Decrease Relative to Output For $\alpha=1$', size=15)
  alpha_plot.tick_params(labelsize=15)
  plt.tight_layout()
  plot_path = os.path.join(outdir, 'multi_objective_eval.pdf')
  alpha_plot.get_figure().savefig(plot_path)

  # clean up the temporary directory created for images
  shutil.rmtree(tempdir)

def main(args):
  evaluate_multi(args.database_path, args.userid, args.ncolors, args.impaths, args.outdir)


if __name__ == "__main__":
  parser = ArgumentParser(description='Evaluate the impact of multiobjective optim weight')
  parser.add_argument('database_path', type=str, help='Path to Specimen database file')
  parser.add_argument('userid', type=str, help='User id')
  parser.add_argument('ncolors', type=int, help='target number of colors')
  parser.add_argument('impaths', type=str, help='csv list of image paths')
  parser.add_argument('outdir', type=str, help='Output directory')
  args = parser.parse_args()
  args.impaths = args.impaths.split(',')
  try:
    main(args)
  except Exception as err:
    import pdb
    pdb.post_mortem()