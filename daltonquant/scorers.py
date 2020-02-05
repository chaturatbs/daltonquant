# Compression decisions in DaltonQuant
# are made based on 'scorers'
# This file contains different scorer implementations


import pickle

import colorsys
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.neighbors.kde import KernelDensity
from sklearn.neural_network import MLPRegressor

from specimen.database import queries as s_qs
from specimen import utils as s_utils

# abstract class
class SpecimenScorer(object):
  """
  Interface for any scorer that users specimen data.
  """
  def _fit(self, df):
    raise NotImplementedError("Subclass implementation")

  def _color_metric_matrix(self, color):
    raise NotImplementedError("Subclass implementation")

  def sorted_candidates(self, color_cts, alpha):
    colors = color_cts[list('rgb')].values
    metric_matrix = self._color_metric_matrix(colors)
    # remove elements in diagonal, since those are just comparisons
    # to themselves
    metric_matrix_diag_indices = np.diag_indices(metric_matrix.shape[0])
    # remove by making them nan
    metric_matrix[metric_matrix_diag_indices] = np.nan
    max_metric = np.nanmax(metric_matrix)
    assert(max_metric != 0)
    norm_metric_matrix = metric_matrix / max_metric
    color_cts['norm_ct'] = color_cts['ct'] / color_cts['ct'].max()
    norm_cts = dict(list(zip(color_cts[list('rgb')].apply(tuple, axis=1), color_cts['norm_ct'])))
    pairs = []
    ncolors = len(colors)
    for i in range(0, ncolors):
      for j in range(0, i):
        # normalized and weighted count of color that would be replaced
        norm_ct_color_i = norm_cts[tuple(colors[i])]
        norm_ct_color_j = norm_cts[tuple(colors[j])]
        weighted_norm_ct_color = (1 - alpha) * max(norm_ct_color_i, norm_ct_color_j)
        # keep color with largest number of pixels
        old, new = (j, i) if (norm_ct_color_j < norm_ct_color_i) else (i, j)
        interpolated_metric = alpha * norm_metric_matrix[i, j] + weighted_norm_ct_color
        entry = (colors[old], colors[new], interpolated_metric)
        pairs.append(entry)
    # larger first
    s = sorted(pairs, key=lambda x: -x[-1])
    return s

  def fit(self, df=None):
    """ Fit estimator based on userid from specimen database """
    try:
      df = self._prepare_data(df)
      print("Fitting with history: nobs=%d" % (df.shape[0]))
      self._fit(df)
    except:
     self._load_predefined()

  def _prepare_data(self, df):
    if df is None:
      if self.userid is None:
        print("Retrieving a sample user id, none provided")
        self.userid = self._get_userid()
      print("Populating data for user id %s" % self.userid)
      df = self._get_user_data()
    return df

  def _load_predefined(self):
    print("Failed to retrieve specimen data for user indicated")
    print("Loading pre-calculated scorer")
    # predefined scorer
    with open(self.default_predef_path, 'r') as default_predef_file:
      self.scorer = pickle.load(default_predef_file)
    self.userid = self.default_userid

  def _get_userid(self, n=0):
    """ Get userid for the nth user, when sorted by descending number of selections """
    db = s_qs.SpecimenQueries(database_path=self.database_path)
    cts_by_users = db.execute_adhoc("""
      select selectionevents.uniqueId as userid, count(*) as ct 
      from selectionevents inner join users on selectionevents.userid = users.id
      where userid > 1 group by selectionevents.uniqueId
      """
      )
    cts_by_users = cts_by_users.sort_values('ct', ascending=False)
    return cts_by_users.iloc[n]['userid']

  def _get_user_data(self):
    db = s_qs.SpecimenQueries(database_path=self.database_path)
    df = db.execute_adhoc("""
      select users.uniqueid as userid,
        specimen_h,
        target_h,
        specimen_r, specimen_g, specimen_b,
        target_r, target_g, target_b,
        specimen_lab_l, specimen_lab_a, specimen_lab_b,
        target_lab_l, target_lab_a, target_lab_b,
        correct
        from selectionevents inner join users 
        on selectionevents.userid = users.id
        where users.uniqueId = '%s'
        """ % self.userid)
    # make sure correct is boolean, sqlite represents as integer, which messes pandas up 
    # sometimes when using as a predicate for row selection
    df['correct'] = df['correct'] == 1
    return df


class AbstractDensityScorer(SpecimenScorer):
  def _prepare_data(self, df=None):
    """ Add in perceptual distance using LAB space """
    df = super(AbstractDensityScorer, self)._prepare_data(df)
    # add perceptual distance
    df = self._add_perceptual_distance(df)
    return df

  def _distance(self, df, coords1, coords2):
    """ Euclidean distance """
    return np.sqrt(((df[coords1].values - df[coords2].values) ** 2).sum(axis = 1))

  def _add_perceptual_distance(self, df):
    lab = s_utils.prefix('lab_', list('lab'))
    specimen_lab = s_utils.prefix('specimen_', lab)
    target_lab = s_utils.prefix('target_', lab)
    df = df.copy()
    df['distance'] = self._distance(df, specimen_lab, target_lab)
    return df

  def _score_samples(self, dists, colors1, colors2):
    raise Exception("Implemented in subclass")

  def _color_metric_matrix(self, colors):
    lab = s_utils.mat_rgb_to_lab(colors / 255.0)
    flat_dists = distance.pdist(lab, metric='euclidean')
    # vector of colors on each side
    # fetch colors based on indices for flat_dists above
    ncolors = len(lab)
    # TODO: there must be a better way
    ix1 = np.repeat(np.arange(ncolors), np.arange(ncolors - 1, -1, -1))
    ix2 = np.hstack([np.arange(x, ncolors) for x in np.arange(ncolors) + 1])
    # density estimates
    flat_dens = np.exp(self._score_samples(flat_dists, colors[ix1], colors[ix2]))
    # make densities into matrix and negate to sort correctly
    return distance.squareform(flat_dens)


class GlobalKDE(AbstractDensityScorer):
  """ Global KDE of perceptual distances in user color confusions """
  def __init__(self, userid, database_path):
    self.database_path = database_path
    self.userid = userid
    self.kde = None
    self.default_userid = 52386
    self.default_predef_path = 'resources/global_kde'
    self.scorer_method = 'global kde'

  def _score_samples(self, dists, colors1, colors2):
    # ignores colors, since don't care about hue bucket
    return self.kde.score_samples(dists.reshape(-1, 1))

  def _fit(self, df, kernel='gaussian'):
    """ Estimate density for errors as a function of perceptual distance """
    df = df.copy()
    errors = df[~df['correct']]
    kde = KernelDensity(kernel=kernel)
    # this may take a bit if it is a large sample
    kde.fit(errors['distance'].values.reshape(-1, 1))
    self.kde = kde

class HueKDE(AbstractDensityScorer):
  """ Hue-based KDE of perceptual distances in user color confusions """
  def __init__(self, userid, database_path, min_obs=100):
    self.database_path = database_path
    self.userid = userid
    self.hue_kdes = {}
    self.min_obs = min_obs
    self.default_userid = 52386
    self.default_hue_kdes_path = 'resources/hue_kdes'
    self.default_global_kde_path = 'resources/global_kde'
    self.scorer_method = 'hue kde'

  def _load_predefined(self):
    print("Failed to retrieve specimen data for user indicated")
    print("Loading pre-calculated estimator")
    with open(self.default_hue_kdes_path, 'r') as hue_kdes_file:
      self.hue_kdes = pickle.load(hue_kdes_file)
    with open(self.default_global_kde_path, 'r') as global_kde_file:
      self.kde = pickle.load(global_kde_file)
    self.userid = self.default_userid

  def _fit(self, df, kernel='gaussian', plot=False, **kwargs):
    """ Estimate density for errors as a function of perceptual distance """
    df = df.copy()
    # add hue pairs
    df['target_hue'] = s_utils.munsell_buckets(df['target_h'], labels=True)[1]
    df['specimen_hue'] = s_utils.munsell_buckets(df['specimen_h'], labels=True)[1]
    df['hue_pair'] = df[['target_hue', 'specimen_hue']].apply(tuple, axis=1).map(s_utils.munsell_pair_map)
    errors = df[~df['correct']]
    hue_pair_cts = errors.groupby('hue_pair')['target_h'].agg({'ct': np.size}).reset_index()
    enough_cts = hue_pair_cts[hue_pair_cts['ct'] >= self.min_obs]
    for pair in enough_cts['hue_pair']:
      pair_errors = errors[errors['hue_pair'] == pair]
      pair_kde = KernelDensity(kernel=kernel)
      pair_kde.fit(pair_errors['distance'].values.reshape(-1, 1))
      self.hue_kdes[pair] = pair_kde
    # fit a global kde to fall back on
    self.kde = KernelDensity(kernel=kernel)
    self.kde.fit(errors['distance'].values.reshape(-1, 1))

  def _score_samples(self, dists, colors1, colors2):
    # TODO: this is very slow!
    rgb_to_hue = lambda x: colorsys.rgb_to_hsv(*x)[0]
    # Note that we need RGB to be in [0.0, 1.0], so divide by 255
    hue1 = np.apply_along_axis(rgb_to_hue, 1, colors1 / 255.0)
    hue2 = np.apply_along_axis(rgb_to_hue, 1, colors2 / 255.0)
    hue_bucket1 = s_utils.munsell_buckets(hue1, labels=True)[1]
    hue_bucket2 = s_utils.munsell_buckets(hue2, labels=True)[1]
    data = np.vstack([dists, hue_bucket1, hue_bucket2])
    # get appropriate estimator, or fall back on to general one
    get_kde = lambda h1, h2: self.hue_kdes.get(s_utils.munsell_pair_map[(h1, h2)], self.kde)
    # get actual density estimate for each observation
    get_density = np.vectorize(lambda d, h1, h2: get_kde(h1, h2).score_samples(d))
    get_density_vec = np.vectorize(get_density)
    return get_density_vec(dists, hue_bucket1, hue_bucket2)


class AbstractSpecimenTransformer(SpecimenScorer):
  """ Transforms the original color space into a user color space for
  distance calculation """

  def _color_metric_matrix(self, colors):
    # RGB colors need to be converted to LAB colors
    lab_colors = s_utils.mat_rgb_to_lab(colors / 255.0)
    # colors as perceived by user
    user_colors = self._transform(lab_colors)
    # inverse distance
    epsilon = 1e-4
    # add in epsilon to avoid infinity when distance is 0 in transformed space
    # otherwise alpha weighing factor for the number of pixels associated with that
    # color merger will never affect sorting
    return 1.0 / (distance.squareform(distance.pdist(user_colors)) + epsilon)

  def _fit(self, df):
    """ Learn a transformation from target colors to perceived (specimen) colors """
    df = df.copy()
    errors = df[~df['correct']]
    orig_colors = errors[['target_lab_l', 'target_lab_a', 'target_lab_b']]
    perceived_colors = errors[['specimen_lab_l', 'specimen_lab_a', 'specimen_lab_b']]
    self._learn_transform(orig_colors.values, perceived_colors.values)


class LinearTransform(AbstractSpecimenTransformer):
  def __init__(self, userid, database_path):
    self.database_path = database_path
    self.userid = userid
    self.color_transform_matrix = None
    self.scorer_method = 'linear transform (least-squares)'

  def _learn_transform(self, orig_colors, perceived_colors):
    self.color_transform_matrix = np.linalg.lstsq(orig_colors, perceived_colors)[0]

  def _transform(self, orig_colors):
    return np.dot(orig_colors, self.color_transform_matrix)


class NonLinearTransform(AbstractSpecimenTransformer):
  def __init__(self, userid, database_path):
    self.database_path = database_path
    self.userid = userid
    self.nn = MLPRegressor()
    self.scorer_method = 'nonlinear transform (nn)'

  def _learn_transform(self, orig_colors, perceived_colors):
    self.nn.fit(orig_colors, perceived_colors)

  def _transform(self, orig_colors):
    return self.nn.predict(orig_colors)
