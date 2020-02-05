import argparse

from collections import defaultdict
import itertools
import os

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.ion()
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

from specimen.database import queries as s_qs
from ..get_cvd import get_user_data


def extend_stats(df):
    df['pre_quantizer'] = df['method'].map(lambda x: x.split('_')[-1])
    df['scorer'] = df['method'].map(lambda x: x.split('_')[1])
    df['reference_fs'] = (df['pre_quantizer'] == 'pngquant') * df['pngquant_fs'] +\
      (df['pre_quantizer'] == 'mediancut') * df['medquant_fs'] +\
      (df['pre_quantizer'] == 'tinypng') * df['tinypng_fs']
    df['compression_ratio'] = df['reference_fs'] / df['specimen_fs']
    df['fs_decrease'] = df['specimen_fs'] / df['reference_fs'] - 1
    df['orig_compression_ratio'] = df['orig_fs'] / df['specimen_fs']
    df['image_id'] = df['image'].map(lambda x: int(os.path.basename(x)[5:7]))
    return df

def compression_ratio_barplots(input_path, output_path):
  # box plot, x: target ncolors, y: compression ratios, groups: prequantizer
  df = pd.read_csv(input_path)
  df = extend_stats(df)
  fs = df.groupby(['pre_quantizer', 'scorer', 'target_ncolors'])['fs_decrease'].mean().reset_index()
  clean_quantizer = {'mediancut': 'Basic Median Cut', 'pngquant': 'pngquant', 'tinypng': 'TinyPNG'}
  clean_scorer = {'global kde': 'Global KDE', 'hue kde': 'Hue-Specific KDE', 'linear transform (least-squares)': 'Linear Transform', 'nonlinear transform (nn)': 'Non-Linear Transform'}
  fs['pre_quantizer'] = fs['pre_quantizer'].map(clean_quantizer)
  fs['scorer'] = fs['scorer'].map(clean_scorer)
  fs['fs_decrease'] *= 100.0
  p = sns.factorplot(x='target_ncolors', y = 'fs_decrease', hue='scorer', data=fs, kind='bar', col='pre_quantizer', legend=False, sharey=True, size=5)
  # add in hatches
  fresh_hatches = lambda: itertools.cycle(['o', '+', '-', 'x', '\\', '*', '/', 'O', '.'])
  num_colors = len(df['target_ncolors'].unique())
  for ax in p.axes.flatten():
    # resize font
    ax.tick_params(labelsize=15)
    # reset fresh for new subplot
    hatches = fresh_hatches()
    for i, bar in enumerate(ax.patches):
      if i % num_colors == 0:
        hatch = next(hatches)
      bar.set_hatch(hatch)
  p.set_xlabels('Palette Size', size=15)
  p.set_ylabels('% File Decrease vs Reference Image', size=15)
  p.set_titles("Pre-Quantizer: {col_name}", size=15)
  plt.legend(loc='best', prop={'size':14})
  plt.tight_layout()
  p.savefig(output_path)
  
  

def compression_ratio_boxplots(input_path, output_path):
  # box plot, x: target ncolors, y: compression ratios, groups: prequantizer
  df = pd.read_csv(input_path)
  df = extend_stats(df)
  unique_sorted = lambda x: sorted(list(set(x)))
  ncolors = unique_sorted(df['target_ncolors'])
  pre_quantizers = unique_sorted(df['pre_quantizer'])
  n_pre_quantizers = len(pre_quantizers)
  scorers = unique_sorted(df['scorer'])
  plot_data = defaultdict(lambda: [])
  for s in scorers:
    for ncolor in ncolors:
      d_ncolor = df[(df['target_ncolors'] == ncolor) & (df['scorer'] == s)]
      ratios = [d_ncolor[d_ncolor['pre_quantizer'] == pq]['compression_ratio'].values for pq in pre_quantizers]
      plot_data[s].append(ratios)

  fig, axes = plt.subplots(nrows = len(scorers) / 2, ncols = 2, sharey=True, sharex=True, figsize=(10, 8))
  axes = axes.flatten()
  # color map
  cmap = plt.get_cmap('Dark2')
  colors = cmap(np.linspace(0, 1.0, n_pre_quantizers))
  for ai, s in enumerate(scorers):
    ax = axes[ai]
    data = plot_data[s]
    xticks = []
    for i, _ in enumerate(ncolors):
      pos_start = 1 + (n_pre_quantizers + 2) * i
      pos_end = pos_start + n_pre_quantizers
      pos = list(range(pos_start, pos_end))
      bp = ax.boxplot(data[i], positions=pos, widths=0.6)
      xticks.append(np.mean(pos))
      # set colors for pre quantizers
      for j in range(0, n_pre_quantizers):
        color = colors[j]
        plt.setp(bp['boxes'][j], color=color)
        plt.setp(bp['caps'][j * 2], color=color)
        plt.setp(bp['caps'][j * 2 + 1], color=color)
        plt.setp(bp['whiskers'][j * 2], color=color)
        plt.setp(bp['whiskers'][j * 2 + 1], color=color)
        plt.setp(bp['medians'][j], color=color)
    ax.set_xticklabels(['%d' % x for x in ncolors])
    ax.set_xticks(xticks)
    ax.tick_params(labelsize=16)
    ax.set_title('Scorer: %s' % s)
    ax.set_xlim(left=0)
    
  fig.text(0.5, 0.04, 'Number of Colors', ha='center', size=16)
  fig.text(0.04, 0.5, 'Compression Ratio', va='center', rotation='vertical', size=16)
  # legend
  legend_lines = []
  for j in range(0, n_pre_quantizers):
    line = plt.plot([1,1], color=colors[j])[0]
    legend_lines.append(line)
  ax.legend(legend_lines, pre_quantizers)
  for line in legend_lines:
    line.set_visible(False)
  
  fig.savefig(output_path)
  
def cvd_user_accuracy(db, input_path, output_path):
  df = pd.read_csv(input_path)
  userids = df['userid'].values
  accs = [get_user_data(db, x)['correct'].mean() for x in userids]
  accs = np.array(accs) * 100.0
  fig, ax = plt.subplots(1, figsize=(8, 7.5))
  # ugly hack to avoid sns trying to convert string ids to number
  pcvd_id = ['User %02d' % x for x in range(1, len(userids) + 1)]
  summary_df = pd.DataFrame(list(zip(pcvd_id, accs)), columns=['pcvd_id', 'acc'])
  ax = sns.barplot(x='acc', y='pcvd_id', color=sns.color_palette('deep')[0], data=summary_df)
  ax.set_xlim(left=min(accs) - 5, right=max(accs) + 5)
  # mean specimen accuracy
  correct_cts = db.execute_adhoc('select count(*) as ct from selectionevents group by correct')
  mean_acc = (correct_cts['ct'] / correct_cts['ct'].sum()).iloc[1] * 100.0
  ymin, ymax = ax.get_ylim()
  ax.vlines(mean_acc, ymin, ymax, label='Aggregate Specimen Accuracy', color = sns.color_palette('deep')[2], linestyle='dashed', linewidth=4)
  ax.set_ylabel('')
  ax.set_xlabel('Specimen Accuracy', fontsize=16)
  ax.tick_params(labelsize=16)
  # locs, labels = plt.xticks()
  # plt.setp(labels, rotation=90)
  plt.legend(bbox_to_anchor=(1.0, 1.1), prop={'size': 15})
  fig.savefig(output_path)
  

def print_ratio_summary(info):
  if not isinstance(info, pd.DataFrame):
    info = pd.read_csv(info)
  df = extend_stats(info)
  # average compression ratio and std error
  # then this same thing but for the aggregate, print out
  stats = ['mean', 'std', 'min', 'max', 'median']
  by_user_summary = df.groupby(['userid', 'pre_quantizer', 'scorer', 'target_ncolors'])['compression_ratio'].agg(stats)
  overall_summary = df.groupby(['pre_quantizer', 'scorer', 'target_ncolors'])['compression_ratio'].agg(stats)
  summary_header = lambda x: "==================== %s =================" % x

  print(summary_header("By user"))
  print(by_user_summary)

  print(summary_header("Overall"))
  print(overall_summary)


def avg_ratios_and_decreases(info):
  if not isinstance(info, pd.DataFrame):
    info = pd.read_csv(info)
  df = extend_stats(info)
  df = df[df['pre_quantizer'].isin(['tinypng', 'pngquant'])].groupby(['userid', 'pre_quantizer', 'scorer'])[['fs_decrease', 'compression_ratio']].mean().reset_index()
  print(df.groupby('pre_quantizer')['fs_decrease'].agg(['min', 'max']))
  print(df.groupby('pre_quantizer')['compression_ratio'].agg(['min', 'max']))

  


def plot_result_summaries(csv_summary, output_dir):
  df = pd.read_csv(csv_summary)
  df['reduction_orig'] = (1 - df['specimen_fs'] / df['orig_fs'].astype(float)) * 100
  df['reduction_pngquant'] = (1 - df['specimen_fs'] / df['pngquant_fs'].astype(float)) * 100
  df['reduction_tinypng'] = (1 - df['specimen_fs'] / df['tinypng_fs'].astype(float)) * 100
  images = list(set(df['image']))
  for image in images:
    df0 = df[(df['image'] == image)]
    fig, plots = plt.subplots(2, ncols=1)
    size_plot = plots[0]
    quality_plot = plots[1]
    # size calculation
    df0.plot(x='target_ncolors', y='reduction_pngquant', label='Rel. to pnquant', kind='scatter', color='blue', marker='o', ax=size_plot)
    df0.plot(x='target_ncolors', y='reduction_tinypng', label='Rel. to tinypng', kind='scatter', color='blue', marker='>', ax=size_plot)
    size_plot.legend(loc='best')
    size_plot.axhline(y=0)
    size_plot.set_xlabel('Number of Colors', visible=False)
    size_plot.set_ylabel('% Reduction in File Size')
    # figure level stuff
    fig.suptitle('Results for %s' % image, fontsize=14)
    image_name = image.split('/')[1].split('.')[0]
    plot_name = '%s_size_results.png' % image_name
    plot_path = os.path.join(output_dir, plot_name)
    fig.savefig(plot_path)
    plt.close()
    
  
def tabulate_index_distances(im, gen_neighbors):
  assert(im.mode == 'P')
  pixels = im.load()
  pixels = np.array(pixels)
  nrows, ncols = im.size
  dists = []
  for i in range(nrows):
    for j in range(ncols):
      p1 = pixels[i, j]
      for neighbor in gen_neighbors(i, j):
        iprime, jprime = neighbor
        if iprime >= 0 and iprime < nrows and jprime >= 0 and jprime < ncols:
          p2 = pixels[iprime, jprime]
          dists.append(np.abs(p1 - p2))
  return pd.DataFrame({'dists': dists})
  
def average_absolute_filesizes(info, output_path):
    if not isinstance(info, pd.DataFrame):
        info = pd.read_csv(info)
    df = extend_stats(info)
    summary = df.groupby(['target_ncolors', 'pre_quantizer'])[['reference_fs', 'specimen_fs']].mean() / 1e3
    summary = summary.reset_index()
    summary['pre_quantizer'] = summary['pre_quantizer'].map({'mediancut': 'Basic Median Cut', 'tinypng': 'TinyPNG', 'pngquant': 'pngquant'})
    summary = summary.rename(columns={'target_ncolors': 'Palette Size', 'pre_quantizer': 'Pre-Quantizer', 'reference_fs': 'Reference (kB)', 'specimen_fs': 'DaltonQuant (kB)'})
    summary.to_latex(output_path, index=False, float_format='%.2f')
    
     
  
  
if __name__ == "__main__":
  description = """
  Analyze results. Wraps a couple of utility plots etc
  """

  available_actions = {
    'ratio_boxplot': lambda args: compression_ratio_boxplots(args.input_path, args.output_path), 
    'ratio_summary': lambda args: print_ratio_summary(args.input_path), 
    'compression_ratio_barplots': lambda args: compression_ratio_barplots(args.input_path, args.output_path), 
    'avg_ratios_and_decreases': lambda args: avg_ratios_and_decreases(args.input_path), 
    'cvd_user_accuracy': lambda args: cvd_user_accuracy(db, args.input_path, args.output_path),
    'average_absolute_filesizes': lambda args: average_absolute_filesizes(args.input_path, args.output_path)
  }
  available_actions_str = ','.join(list(available_actions.keys()))
  
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('database', type=str, help='Path to Specimen database file')
  parser.add_argument('input_path', type=str, help="Path to results relevant for given function")
  parser.add_argument('output_path', type=str, help="Path to save down plot/result relevant for given function")
  parser.add_argument('action', type=str, help='One of [%s]' % available_actions_str)
  args = parser.parse_args()

  db = s_qs.SpecimenQueries(database_path=args.database)

  
  if not args.action in available_actions:
    raise ValueError("Undefined action: %s, must be one of [%s]" % (args.action, available_actions_str))
  
  available_actions.get(args.action)(args)

