from argparse import ArgumentParser
import os

from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

from ..scorers import LinearTransform, NonLinearTransform


def compute_distances(database_path, users, output_dir, user_names=None):
  if user_names is None:
    user_names = list(range(len(users)))
  
  user_name_dict = dict(zip(users, user_names))
  acc = []
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
      kf = KFold(n_splits=10, random_state=0, shuffle=False)
      distance_ratios = np.array([], dtype=float)
      for train_ix, test_ix in kf.split(range(df.shape[0])):
        print("Evaluating fold")
        train_target_vals = target_vals[train_ix]
        train_specimen_vals = specimen_vals[train_ix]
        model.fit(train_target_vals, train_specimen_vals)
      
        test_target_vals = target_vals[test_ix]
        test_specimen_vals = specimen_vals[test_ix]
      
        # remove from test values anything we've actually already seen (colors are repeated)
        observed = np.vstack([train_target_vals, train_specimen_vals])
        seen_target = np.array([(e == observed).all(axis=1).any() for e in test_target_vals])
        seen_specimen = np.array([(e == observed).all(axis=1).any() for e in test_specimen_vals])
        seen = seen_target | seen_specimen
    
        test_target_vals = test_target_vals[np.where(~seen)]
        test_specimen_vals = test_specimen_vals[np.where(~seen)]
      
        if len(test_target_vals) > 0:
          test_orig_distances = np.sqrt(np.sum(np.square(test_target_vals - test_specimen_vals), axis=1))
      
          test_transform_target_vals = model.predict(test_target_vals)
          test_transform_specimen_vals = model.predict(test_specimen_vals)
      
          test_transform_distances =  np.sqrt(np.sum(np.square(test_transform_target_vals - test_transform_specimen_vals), axis=1))
      
          distance_ratios = np.append(distance_ratios, test_transform_distances / test_orig_distances - 1)
        
      
      result = (user_name_dict[u], model_name, len(distance_ratios), distance_ratios.mean(), distance_ratios.std())
      acc.append(result)

  columns = ['user', 'transform', 'n', 'mean_val', 'std_val']
  results_df = pd.DataFrame(acc, columns=columns)
  results_df['val'] = ["%.2f (+/-%.2f)" % (m, s) for m,s in zip(results_df.mean_val, results_df.std_val)]
  clean_columns = {
    'user': 'PCVD User',
    'transform': 'Transformation',
    'val': 'Mean Change Distance (+/- Std)'
  }
  results_df = results_df[['user', 'transform', 'val']].rename(columns=clean_columns)
  output_path = os.path.join(output_dir, 'transformed_distance_changes.tex')
  results_df.to_latex(output_path, index=False)
  
def main(args):
  database_path = args.database
  userids = [u.strip() for u in args.userids.split(',')]
  output_dir = args.output_dir
  user_names = args.names.split(',') if args.names else None
  compute_distances(database_path, userids, output_dir, user_names=user_names)

if __name__ == "__main__":
  parser = ArgumentParser(description='Evaluate the impact of transformations on the distances across color spaces for Specimen selections')
  parser.add_argument('database', type=str, help='Path to Specimen database file')
  parser.add_argument('userids', type=str, help='csv list of user ids to query')
  parser.add_argument('output_dir', type=str, help='Directory to output figure')
  parser.add_argument('--names', type=str, help='csv list of user names to display in figure')
  args = parser.parse_args()
  try:
    main(args)
  except Exception as err:
    import pdb
    pdb.post_mortem()


