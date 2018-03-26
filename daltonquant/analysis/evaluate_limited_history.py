from argparse import ArgumentParser
import itertools
import glob
import os
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

from ..compressors import PngQuant
from ..scorers import LinearTransform, NonLinearTransform
from ..utils import pil as p_utils


# get data
def mean_rss(database_path, users, output_dir, user_names=None):
  #users = pd.read_csv('ext_users/cvd_users_info.csv')['userid']
  if user_names is None:
    user_names = range(1, len(users) + 1)
  user_name_map = dict(zip(users, user_names))
  pcts = np.arange(0.1, 0.9, 0.1)
  iters = 10

  acc = []
  np.random.seed(1)
  for u in users:
    df = LinearTransform(u, database_path=database_path)._get_user_data()
    # only care about errors
    df = df[~df.correct]
    target_vals = df[['target_lab_l', 'target_lab_a', 'target_lab_b']].values
    specimen_vals = df[['specimen_lab_l', 'specimen_lab_a', 'specimen_lab_b']].values
    models = {
      'Linear': LinearRegression(fit_intercept=False),
      'Non-Linear': MLPRegressor()
    }
    for model_name, model in models.items():
      # repeat it X times
       for pct in pcts:
         repeated = 0
         while repeated < iters:
           print "Model: %s, User: %s, Pct: %.2f [%d/%d]" % (model_name, u, pct, repeated, iters)
           train_target_vals, _, train_specimen_vals, _ = train_test_split(target_vals, specimen_vals, test_size=1 - pct)
           model.fit(train_target_vals, train_specimen_vals)

           # remove from test values anything we've actually already seen (colors are repeated)
           observed = np.vstack([train_target_vals, train_specimen_vals])
           seen_target = np.array([(e == observed).all(axis=1).any() for e in target_vals])
           seen_specimen = np.array([(e == observed).all(axis=1).any() for e in specimen_vals])
           seen = seen_target | seen_specimen

           remaining_target_vals = target_vals[np.where(~seen)]
           remaining_specimen_vals = specimen_vals[np.where(~seen)]
           # compute residual sum of squares
           if len(remaining_target_vals) > 0:
             max_pct = 1 - pct
             observed_pct = remaining_target_vals.shape[0] / float(target_vals.shape[0])
             predicted_specimen_vals = model.predict(remaining_target_vals)
             diffs = remaining_specimen_vals - predicted_specimen_vals
             mean_rss = np.mean(np.sum(np.power(diffs, 2), axis=0))
             acc.append((user_name_map[u], model_name, pct, repeated, mean_rss))
             repeated += 1

  columns = ['user',  'model', 'pct', 'iter_id', 'mean_rss']
  df = pd.DataFrame(acc, columns=columns)
  summary_df = df.groupby(['user', 'model', 'pct'])[['mean_rss']].mean().reset_index()
  output_summary_path = os.path.join(output_dir, 'mean_rss_reduced_history.csv')
  summary_df.to_csv(output_summary_path, index=False)
  sample_summary_df = summary_df[summary_df['user'].isin([1,2,3])]
  rss_plots = sns.FacetGrid(data=sample_summary_df, col='model', size=5)
  rss_plots.map(sns.pointplot, 'pct', 'mean_rss', 'user', palette='deep')
  plt.legend(loc='best', title='PCVD User')
  plt.ylim(bottom=0.0)
  rss_plots.set_xlabels('Fraction of Specimen Confusion History', size=15)
  rss_plots.set_ylabels('Mean RSS on unseen (target, specimen) pairs', size=15)
  rss_plots.set_titles('{col_name} Transform', size=15)
  for ax in rss_plots.axes.flatten():
    ax.tick_params(labelsize=15)
  plt.tight_layout()
  rss_plot_path = os.path.join(output_dir, 'mean_rss_reduced_history.pdf')
  rss_plots.savefig(rss_plot_path)
  plt.close('all')

def index_in_new_list(orig_list, new_list):
  # find the index for each element
  # TODO: this is pretty wasteful....
  idx = [np.where(np.all(e == new_list, axis=1))[0][0] for e in orig_list]
  return idx

def rank_correlation(database_path, user, impaths, output_dir):
  # for a single user
  pcts = np.arange(0.1, 1.1, 0.1)
  # for reproducibility
  np.random.seed(1)
  # repeat each measurement n_iters times
  n_iters = 10
  # compare ranks for top n color candidates
  top_n_candidates = 200
  # data for this user
  df = LinearTransform(user, database_path=database_path)._get_user_data()
  # only errors
  df = df[~df.correct]
  transforms = {'linear': LinearTransform, 'non-linear': NonLinearTransform}
  acc = []
  for impath in impaths:
    print "Image Path: %s" % impath
    im_name = int(os.path.basename(impath)[5:7])
    temp = NamedTemporaryFile(suffix='.png')
    im = PngQuant().compress(impath, outpath=temp.name, ret_im=True)
    im = p_utils.to_rgb(im)
    color_cts = p_utils.count_colors(im)
    for transform_name, transform in transforms.items():
      print "Transform: %s" % transform_name
      # fit with full data
      orig_scorer = transform(user, database_path=database_path)
      # set random state for nonlinear
      # to get same training and compare appropriately
      if transform == NonLinearTransform:
        orig_scorer.nn.random_state = 1
      orig_scorer.fit(df)
      # only consider the top N candidates
      orig_sorted_candidates = orig_scorer.sorted_candidates(color_cts, 1.0)[:top_n_candidates]
      orig_sorted_candidates = np.array(map(lambda x: x[:-1], orig_sorted_candidates)).reshape(-1, 6)
      orig_ranking = np.arange(len(orig_sorted_candidates))
      for pct in pcts:
        for i in range(n_iters):
          print "User %s with history fraction %f [%d/%d]" % (user, pct, i + 1, n_iters)
          df_sample = df.sample(frac=pct)
          # rank colors with new_version
          scorer = transform(user, database_path=database_path)
          if transform == NonLinearTransform:
            scorer.nn.random_state = 1
          scorer.fit(df_sample)
          sorted_candidates = scorer.sorted_candidates(color_cts, 1.0)
          sorted_candidates = np.array(map(lambda x: x[:-1], sorted_candidates)).reshape(-1, 6)
          new_ranking = index_in_new_list(orig_sorted_candidates, sorted_candidates)
          rho = spearmanr(orig_ranking, new_ranking)
          acc.append((im_name, transform_name, pct, i, new_ranking, rho.correlation, rho.pvalue))

  df = pd.DataFrame(acc, columns=['im_name', 'transform_name', 'pct', 'iter', 'new_ranking', 'rho', 'pvalue'])
  df_path = os.path.join(output_dir, 'rank_correlation_reduced_history.csv')
  df.to_csv(df_path, index=False)
  summary_df = df.groupby(['im_name', 'transform_name', 'pct'])[['rho']].mean().reset_index()
  summary_df['clean_transform_name'] = summary_df['transform_name'].map({'linear': 'Linear', 'non-linear': 'Non-Linear'})
  
  transform_names = {'linear': 'Linear', 'non-linear': 'Non-Linear'}
  color_palette = sns.color_palette("hls", 24)
  for name in transform_names.keys():
      subset_df = summary_df[summary_df['transform_name'] == name]
      subset_plot = sns.pointplot(data=subset_df, x='pct', y='rho', hue='im_name', palette=color_palette)
      subset_plot.set_title('%s Transformation' % transform_names[name], size=20)
      subset_plot.set_xlabel('Fraction of Specimen Confusion History', size=15)
      subset_plot.set_ylabel("Spearman's rho over top 200 color merger candidates", size=15)
      subset_plot.tick_params(labelsize=15)
      subset_plot.set_ylim(bottom=0.3, top=1.0)
      plt.legend(title='Kodak Image', loc='best', ncol=4, fontsize=15)
      plt.tight_layout()
      plot_path = os.path.join(output_dir, 'rank_correlation_reduced_history-%s.pdf' % name)
      subset_plot.get_figure().savefig(plot_path)
      plt.close('all')


def main(args):
  args.names = args.names.split(',') if args.names else None
  args.users = args.users.split(',')
  args.input_images = args.input_images.split(',')
  mean_rss(args.database, args.users, args.output_dir, user_names=args.names)
  rank_correlation(args.database, args.users[0], args.input_images, args.output_dir)


if __name__ == "__main__":
  parser = ArgumentParser(description='Evaluate the impacts of limited history on compression')
  parser.add_argument('database', type=str, help='Path to Specimen database file')
  parser.add_argument('users', type=str, help='csv list of user unique ids (first id used for rank correlation analysis)')
  parser.add_argument('input_images', type=str, help='csv list of input images')
  parser.add_argument('output_dir', type=str, help='Output directory')
  parser.add_argument('--names', type=str, help='User names')
  args = parser.parse_args()
  try:
    main(args)
  except Exception as err:
    import pdb
    pdb.post_mortem()
