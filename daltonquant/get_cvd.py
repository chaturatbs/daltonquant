# DaltonQuant relies on extracting information about color vision deficient users'
# perception to drive its quantization decisions. To do so, it uses
# Specimen data. This file implements a lot of the color vision deficient
# querying logic needed.

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from specimen.plot import colorplots
import specimen.database.queries as s_qs
import specimen.utils as s_utils


# at least min_obs selections
def get_cvd(db, n, min_obs=1000):
  """ get n users with high cvd and high number of observations """
  enough_selections_query = """
    select users.uniqueid as userid 
    from 
    selectionevents inner join users on selectionevents.userid = users.id
    where userid > 1 
    group by userid having count(*) > %d
  """
  userids = db.execute_adhoc(enough_selections_query % min_obs)['userid'].values
  
  userids_str = ['"%s"' % _id for _id in userids]
  return compute_cvd_info(db, userids)
  
def compute_cvd_info(db, userids):
  userids = ['"%s"' % _id for _id in userids]
  userids_str = ','.join(userids)
  relevant_fields_query = """
    select 
    users.uniqueid as userid,
    target_h, specimen_h,
    target_lab_l, target_lab_a, target_lab_b,
    specimen_lab_l, specimen_lab_a, specimen_lab_b,
    correct
    from 
    selectionevents inner join users on selectionevents.userid = users.id
    where users.uniqueid in (%s) """
  return  db.execute_adhoc(relevant_fields_query % userids_str)
  

def wrong_hue(df):
  """ Fraction of selections in wrong hue bucket vs target (all selections, not just wrong) """
  df['target_hue_bucket'] = s_utils.munsell_buckets(df['target_h'], labels=True)[1]
  df['specimen_hue_bucket'] = s_utils.munsell_buckets(df['specimen_h'], labels=True)[1]
  df['wrong_hue'] = df['target_hue_bucket'] != df['specimen_hue_bucket']
  return df.groupby('userid')[['wrong_hue']].mean()


def get_LAB_distance(df):
  """ Add distance in CIE LAB """
  space = list('lab')
  get_values = lambda df, cat:df[['%s_lab_%s' % (cat, dim) for dim in space]].values
  diffs = get_values(df, 'specimen') - get_values(df, 'target')
  return np.sqrt(diffs ** 2).sum(axis=1)


def avg_distance(df):
  """ Average distance of all selections (not just wrong) """
  df['dist'] = get_LAB_distance(df)
  return df.groupby('userid')[['dist']].mean()


def rank_cvd(df, alpha):
  # between 0 and 1
  metric1 = wrong_hue(df)
  # unbounded, TODO: scale this?
  metric2 = avg_distance(df)
  metrics = metric1.merge(metric2, how='inner', left_index=True, right_index=True)
  normalized_metrics = (metrics - metrics.mean()) / metrics.std()
  normalized_metrics['cvd_score'] = alpha * normalized_metrics.wrong_hue + (1 - alpha) * normalized_metrics.dist
  nobs = df.groupby('userid')[['correct']].count().rename(columns={'correct':'nobs'})
  # accuracies
  accs = df.groupby('userid')[['correct']].mean().rename(columns={'correct': 'accuracy'})
  normalized_metrics = normalized_metrics.merge(nobs, how='inner', left_index=True, right_index=True)
  normalized_metrics = normalized_metrics.merge(accs, how='inner', left_index=True, right_index=True).reset_index()
  return normalized_metrics.sort_values(['cvd_score', 'nobs'], ascending=False)

def get_user_data(db, userid):
  cvd_data_query = """
  select 
  users.uniqueid as userid, 
  target_r, target_g, target_b, 
  specimen_r, specimen_g, specimen_b, 
  correct, target_h, specimen_h,
  target_lab_l, target_lab_a, target_lab_b,
  specimen_lab_l, specimen_lab_a, specimen_lab_b
  from 
  selectionevents inner join users on selectionevents.userid = users.id
  where users.uniqueid = "%s"
  """
  selections = db.execute_adhoc(cvd_data_query % userid)
  selections['target_hue_bucket'] = s_utils.munsell_buckets(selections['target_h'], labels=True)[1]
  selections['specimen_hue_bucket'] = s_utils.munsell_buckets(selections['specimen_h'], labels=True)[1]
  selections['hue_pair'] = selections[['target_hue_bucket', 'specimen_hue_bucket']].apply(tuple, axis=1).map(s_utils.munsell_pair_map)
  selections['dist'] = get_LAB_distance(selections)
  # make sure we type cast to boolean, to avoid issues with pandas and integer rep (sqlite) of boolean
  selections['correct'] = (selections['correct'] == 1)
  return selections


def plot_cvd_diagnostics(db, userid, nswatches=20, combined_plots=False):
  selections = get_user_data(db, userid)
  # diagnostics plots, combine them if necessary
  if combined_plots:
    diagnostics_fig, diagnostics_plots = plt.subplots(3, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5)
  else:
    diagnostics_fig = None
    diagnostics_plots = []
    for _ in [1] * 3:
      diagnostics_plots.append(plt.subplots(1)[1])
  
  # accuracy by hue bucket for this user
  hue_acc_plot = diagnostics_plots[0]
  acc_by_hue = selections.groupby('target_hue_bucket')['correct'].mean()
  acc_by_hue.plot(x='target_hue_bucket', y='acc', kind='bar',  ax=hue_acc_plot)
  hue_acc_plot.set_xlabel('Munsell Hue')
  hue_acc_plot.set_ylabel('Accuracy')
  hue_acc_plot.set_title('Hue Accuracy')
  
  # histogram of hue bucket pairs (order indep)
  hue_pair_error_cts = selections[~selections['correct']].groupby('hue_pair')['userid'].agg({'ct': len})
  hue_pair_error_cts = hue_pair_error_cts / hue_pair_error_cts['ct'].sum()
  hue_pair_histo_plot = diagnostics_plots[1]
  hue_pair_error_cts.plot(kind='bar', ax=hue_pair_histo_plot, label=None)
  hue_pair_histo_plot.set_xlabel('Munsell Hue Pair (Order-independent)')
  hue_pair_histo_plot.set_ylabel('Fraction of Errors')
  hue_pair_histo_plot.set_title('Distribution of Errors Across Hue Pairs')
  hue_pair_histo_plot.legend().remove()
  
  # swatches, sorted by perceptual distance
  selections = selections.sort_values('dist', ascending=False)
  swatch_cols = ['target_h', 'target_r', 'target_g', 'target_b', 'specimen_h', 'specimen_r', 'specimen_g', 'specimen_b']
  sample_user_colors = selections[swatch_cols].drop_duplicates()
  # top n by distance
  sample_user_colors = sample_user_colors.head(nswatches)
  # sort by hues (target and then specimen)
  sample_user_colors = sample_user_colors.sort_values(['target_h', 'specimen_h'], ascending=True)
  rgb_cols = [col for col in sample_user_colors if not col in ['target_h', 'specimen_h']]
  sample_user_colors = sample_user_colors[rgb_cols]
  _, swatches_plot = plt.subplots(1, figsize=(4, 6))
  diagnostics_plots[2] = swatches_plot
  colorplots.color_pairs_plot(*(sample_user_colors.values.T), groups=1, ax=swatches_plot)
  swatches_plot.set_title('Sample Distant Confusions')
  
  if combined_plots:
    diagnostics_fig.suptitle('Diagnostics Plot for Potential CVD userid=%s' % userid, fontsize=16)
    return {'combined' : diagnostics_fig}
  else:
    return {'swatches': swatches_plot.get_figure(), 'hue_acc': hue_acc_plot.get_figure(), 'hue_pair_histo': hue_pair_histo_plot.get_figure()}


def main(args):
  outdir = args.outdir
  # global database
  db = s_qs.SpecimenQueries(database_path=args.database)
  if not args.userids:
    cvd_user_info = get_cvd(db, args.n_users, min_obs=args.min_obs)
    cvd_user_info = rank_cvd(cvd_user_info, alpha=args.alpha).head(args.n_users)
  else:
    args.userids = args.userids.split(',')
    cvd_user_info = compute_cvd_info(db, args.userids)
    cvd_user_info = rank_cvd(cvd_user_info, alpha=args.alpha)
  
  # save a copy of the info to a folder for later reference
  table_path = os.path.join(outdir, 'cvd_users_info.csv')
  cvd_user_info.to_csv(table_path, index=False)
  # just the user ids
  cvd_user_ids = cvd_user_info['userid'].values
  # produce diagnostic plots for each user
  if args.plot:
    for user in cvd_user_ids:
      print("Plotting %s" % user)
      ps = plot_cvd_diagnostics(db, user, combined_plots=args.combine)
      for p_nm, p in ps.items():
        plot_path = os.path.join(outdir, 'cvd_diagnostics_%s_%s.pdf' % (p_nm, user))
        p.savefig(plot_path)
      plt.close('all')
  
  print(args.sep.join(map(str, cvd_user_ids)))

  

  
if __name__ == '__main__':
  description = """ 
  Extract users with a high heuristic score for color vision deficiency. The
  score is a weighted average of the fraction of selections that a user makes where
  the hue bucket (munsell) is different from the target, and the average perceptual distance
  of their selections.
  These values are normalized w.r.t to the qualifiying population (at least N observations), and then combined.
  """
  
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('database', type=str, help='Path to specimen database file')
  parser.add_argument('-u', '--userids', type=str, help='CSV list of user ids')
  parser.add_argument('-n', '--n_users', type=int, help='Number of user ids to return')
  parser.add_argument('-m', '--min_obs', type=int, help='Minimum number of observations for user subpopulation (default=1000)', default=1000)
  parser.add_argument('-a', '--alpha', type=float, help='Weighing factor to combine hue bucket mistakes and LAB distance (default=0.5)', default=0.5)
  parser.add_argument('-s', '--sep', type=str, help='Separator for result list (default=,)', default=',')
  parser.add_argument('-p', '--plot', action='store_true', help='Generate diagnostic plots for each potentially CVD user')
  parser.add_argument('-o', '--outdir', type=str, help='Directory to output diagnostic plot filles', default='.')
  parser.add_argument('-c', '--combine', action='store_true', help='Combine diagnostics plots into single for each user')
  
  args = parser.parse_args()
  if not args.userids and not args.n_users:
    raise Exception("Must provide a list of user ids or a number of users to extract")
  if args.userids:
    print("Ignore min_obs parameters if passed")
  if args.n_users and (not args.alpha or not args.min_obs):
    raise Exception("Must provide alpha and min_obs parameters if providing number of users to extract")

  try:
    main(args)
  except Exception as err:
    import pdb
    pdb.post_mortem()

    