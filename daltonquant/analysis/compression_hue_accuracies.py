# Compute hue accuracies across a set of images, based on images_as_game.py

import argparse
import glob
import os

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import scipy
import seaborn as sns

import specimen.utils as s_utils
from specimen.database import queries as s_qs

from ..get_cvd import get_user_data
from .images_as_game import hue_accuracies


def compute_hue_accuracies(targetdir, specimendir, scorer, userid, ncolors):
  pattern = 'specimen_%s_pngquant_userid_%s_ncolors_%d_*.png' % (scorer, userid, ncolors)
  specimen_files = glob.glob(os.path.join(specimendir, pattern))
  target_files = glob.glob(targetdir + '/*')
  accuracies = hue_accuracies(target_files, specimen_files)
  accuracies['ncolors'] = ncolors
  return accuracies[['hue', 'acc', 'ncolors']]

def plot_hue_accuracies(df):
  df['acc'] *= 100.0
  lineplot = sns.pointplot(x='hue', y='acc', hue='ncolors', data=df)
  lineplot.set_xlabel('Munsell Hue', size=16)
  lineplot.set_ylabel('Accuracy', size=16)
  lineplot.legend(loc='best', title='# colors', ncol=3)
  lineplot.tick_params(labelsize=16)
  plt.subplots_adjust(bottom=0.15)
  return lineplot

def min_max_norm(v):
  min_val = min(v)
  max_val = max(v)
  return (v - min_val) / (max_val - min_val)

def plot_hue_accuracies_lm(df, ncolors, norm=True):
  # scatter plot with linear regression
  subset_df = df[df['ncolors'] == ncolors].copy()
  if norm:
    subset_df['acc'] = min_max_norm(subset_df['acc'])
    subset_df['specimen'] = min_max_norm(subset_df['specimen'])
  ax = sns.lmplot(x='acc',y='specimen',data=subset_df,fit_reg=True)
  extra_info = ' Normalized' if norm else ''
  ax.set_xlabels('Kodak-Based Accuracy' + extra_info, size=15)
  ax.set_ylabels('Specimen Accuracy' + extra_info, size=15)
  # get linear regression R^2
  r_squared = scipy.stats.linregress(subset_df['acc'], subset_df['specimen']).rvalue ** 2
  plt.title(r'%d Target Colors ($R^2$=%.2f)' % (ncolors, r_squared), size=15)
  plt.tick_params(labelsize=15)
  plt.tight_layout()
  return ax

def main(database_path, targetdir, specimendir, scorer, userid, outdir, target_ncolors):
  dfs = [compute_hue_accuracies(targetdir, specimendir, scorer, userid, ncolors) for ncolors in target_ncolors]
  db = s_qs.SpecimenQueries(database_path=database_path)
  images_df = pd.concat(dfs, axis=0)

  observed = get_user_data(db, userid)
  observed = observed.groupby('target_hue_bucket')['correct'].mean().reset_index()
  observed = observed.rename(columns= {'target_hue_bucket': 'hue', 'correct': 'acc'})
  observed['ncolors'] = 'Specimen'
  dfs.append(observed)
  combined = pd.concat(dfs, axis=0)

  print("Correlation of accuracies")
  print(combined.pivot_table(index='hue', columns='ncolors', values='acc').corr())
  p = plot_hue_accuracies(combined)
  plot_path = os.path.join(outdir, '%s_image_hue_accuracies.pdf' % scorer)
  p.get_figure().savefig(plot_path)
  csv_path = os.path.join(outdir, '%s_image_hue_accuracies.csv' % scorer)
  combined.to_csv(csv_path, index=False)

  # plot as scatter plot of points along the kodak and
  # specimen accuracies
  specimen_df = observed[['hue', 'acc']].rename(columns={'acc': 'specimen'})
  images_df = pd.merge(images_df, specimen_df, on='hue', how='left')
  for ncolors in images_df['ncolors'].unique():
    for norm in [True, False]:
      scatter_plot = plot_hue_accuracies_lm(images_df, ncolors, norm=norm)
      norm_str = 'normalized' if norm else 'raw'
      scatter_plot_path = os.path.join(outdir, '%s-%d-image-hue-accuracies-scatter-plot-%s.pdf' % (scorer, ncolors, norm_str))
      scatter_plot.savefig(scatter_plot_path)
  plt.close('all')


if __name__ == "__main__":
  description = """
  Compute hue accuracies based on images compressed. Treat base compressions
  from pngquant as targets, and new compressions as specimens. Perform pixel-wise comparisons.
  """
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('database', type=str, help='Path to Specimen database file')
  parser.add_argument('targetdir', type=str, help='Dir with reference pngquant images')
  parser.add_argument('specimendir', type=str, help='Dir with incr. compressed images')
  parser.add_argument('scorer', type=str, help='Scorer name')
  parser.add_argument('userid', type=str, help='Userid to compute and compare accs')
  parser.add_argument('-o', '--outdir', type=str, help='Output dir', default='.')
  parser.add_argument('-n', '--ncolors', type=str, help='csv list of target number of colors')
  args = parser.parse_args()
  try:
    args.outdir = args.outdir.strip()
    args.ncolors = [int(c) for c in args.ncolors.split(',')] if args.ncolors else [230, 204, 179, 153, 128]
    main(args.database, args.targetdir, args.specimendir, args.scorer, args.userid, args.outdir, args.ncolors)
  except Exception as err:
    import pdb
    pdb.post_mortem()
