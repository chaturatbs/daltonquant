# Reproduce all figures/values etc used in writeup
# All paths relative to top-level directory in project.

help_msg="
DaltonQuant Experiment Replication Script (reproduce.sh)
usage: ./reproduce.sh mode

mode: replication or new
\treplication: Reproduce experiments from DaltonQuant Arxiv Paper
\tnew: Run same experiments with a new set of user ids extracted from Specimen database
"


if [ $# -ne 1 ]
then
  echo "Must provide execution mode: replication or new"
  echo -e "${help_msg}"
  exit 1
fi

if [ $1 == '--help' ] || [ $1 == '-h' ]
then
  echo -e "${help_msg}"
  exit 0
fi

# Resources for compression-related tasks
specimen_database='specimendb'
tiny_png_key=''
tiny_png_folder='generated/tinypng_store/'

# Directories for results/intermediate images
analysis_dir="generated/analysis_results"
compression_dir="generated/compression_results"
compression_one_dir="generated/compression_results_alpha_1"
cvd_users_dir="generated/pcvd_users"
pngquant_kodak="generated/pngquant_kodak"

# Directory and inputs for images
kodak_dir="images/kodak"
images=$(find ${kodak_dir} -type f)
images=$(echo $images | tr ' ' \,)

# Evaluation parameters
iterations=5
min_obs=1000
alpha=0.5
nusers=30

# setup folders necessary
mkdir -p ${analysis_dir}
mkdir -p ${compression_dir}
mkdir -p ${compression_one_dir}
mkdir -p ${cvd_users_dir}
mkdir -p ${pngquant_kodak}
mkdir -p ${tiny_png_folder}

# we computed these before
mode=$1
if [ ${mode} == 'replication' ]
  then
    echo "Running in replication mode. Using previously defined user ids."
    userids="\
506A7DDD727F9ABF6E56D7AED542338025787A46,\
DA12E674A3D79C2F7520A3E062DB8D7CF9D0E473,\
49F54AF2A72F1EF95F16FBB26A8A0F11003688D9,\
63C07CCC1167BC0CE8FF2E4F3A17215106CE6AEA,\
0747FD6A0176F1F2A28F3AD3C9572ADA17B0A0F8,\
C9E8B9111C156B259B1B38F5D051BB56DD0B2528,\
EC591C944EE93C1BFD00667827D683128B797D4C,\
381D7725E88322485895334FC58B56A8EA8F3E54,\
E51BEE48D9CD3AE9A76B0B59008890BF6ACA2F2F,\
00C887CA6FE18559851F18035AD6E55822F29C82,\
5B9E2EEAE92B7A1F78C8741FD71252374D59AD5E,\
2314A2B8B9A78608EFC774B295F175BF1AD92C23,\
9CE30159DB403043F8FBE44A8A7144AF0BEEEA1A,\
58AABF36D267EBD4E837B2B3AABCABD0773B1F30,\
732CBCF8605FA4EF992FF5E5A5F1438C31396B91,\
B1A30B07C4735F1AA6CCB07294DF77725FBF6195,\
D84191886B4F1C10D5D06D4E9393D986CB469173,\
FA25E77637322D60D3ED216D59B2A14B96635059,\
8A5CA4D25AF5AFA3DACA07755C052C9646673695,\
7DB300767BF87F50F3C9C180ADAA4AFF1E9A369B,\
0E5C6223347858DFB7BCBE115D3071D0A9C03CE7,\
B59EA5D7FB4D753B7D74CCC64537E6176F45C965,\
B0D6D196E592FE183217B9890C69C6C12D1AB069,\
AD5C4E8396AE4AF1F98346D4EB176F2EB0886E6D,\
A0776429FBE2E4D7809672D54B1B5AF5F0909BD2,\
95216F4A7AEEB9046D45AB375BE8F228EBCC6A63,\
BB9CABC1E4BBD2E1F46B33C215C020A797339BB4,\
93ED1BE40885ACF4E63483BC03451A4A0E3C1357,\
F70F5B99F9CCB6884FF2D1E1E3B1D984F64AEA97,\
E9BF943069BF2CB66341B2EFC150348C448C989F\
"
user1="506A7DDD727F9ABF6E56D7AED542338025787A46"
three_users='506A7DDD727F9ABF6E56D7AED542338025787A46,DA12E674A3D79C2F7520A3E062DB8D7CF9D0E473,49F54AF2A72F1EF95F16FBB26A8A0F11003688D9'

elif [ ${mode} == 'new' ]
then
  # run new set of experiments
  echo "Running in new experiment mode. Extracting new user ids"
  userids=$(
    python2 -m daltonquant.get_cvd\
     ${specimen_database}\
    --n_users ${nusers}\
    --min_obs ${min_obs}\
    --alpha ${alpha}\
    --outdir ${cvd_users_dir}
   )
  #  if we fail to get new ids, there really isn't much we can do
  if [[ $? -ne 0 ]]
  then
      echo "Unable to retrieve new user ids. Consider increasing RAM."
      exit 1
  fi
  user1=$(echo ${userids} | cut -d\, -f 1)
  three_users=$(echo ${userids} | cut -d\, -f-3)
else
  echo "Unknown execution mode"
  echo -e "${help_msg}"
  exit 1
fi

# generate ids for cvd users
# or take ids and compute values
python2 -m daltonquant.get_cvd ${specimen_database} --userids ${userids} --alpha 0.5 --outdir ${cvd_users_dir} --plot

# CVD User accuracy for selected users
python2 -m daltonquant.analysis.analyze_results\
  ${specimen_database}\
  ${cvd_users_dir}/cvd_users_info.csv\
  "${analysis_dir}/cvd_user_accuracy.pdf"\
  'cvd_user_accuracy'


# performing compression
# actual compression benchmarks
python2 -m daltonquant.analysis.evaluate\
  ${specimen_database}\
  ${images}\
  230,204,179,153,128\
  ${tiny_png_key}\
  ${tiny_png_folder}\
  --userids ${userids}\
  --outdir ${compression_dir}\
  --alpha 0.5


## Evaluating impact of compression
# evaluating results
# bar plots of compresion ratios
python2 -m daltonquant.analysis.analyze_results\
  ${specimen_database}\
  ${compression_dir}/results.csv\
  "${analysis_dir}/bar_plot_summary.pdf"\
  "compression_ratio_barplots"

# text for compression ratios summary
python2 -m daltonquant.analysis.analyze_results\
  ${specimen_database}\
  ${compression_dir}/results.csv\
  '.'\
  ratio_summary > "${analysis_dir}/compression_results_summary.txt"

# average ratios and decreases
python2 -m daltonquant.analysis.analyze_results\
  ${specimen_database}\
  ${compression_dir}/results.csv\
  ''\
  "avg_ratios_and_decreases"

# absolute file size
python2 -m daltonquant.analysis.analyze_results\
  ${specimen_database}\
  ${compression_dir}/results.csv\
  "${analysis_dir}/mean_absolute_file_sizes.tex"\
  "average_absolute_filesizes"


## Validating quantization
# distance of transformations
python2 -m daltonquant.analysis.evaluate_transforms_distance\
  ${specimen_database}\
  ${three_users} \
  ${analysis_dir}\
  --names 1,2,3


# simulated hue accuracies
# quantize baselines with pngquant
python2 -m daltonquant.utils.pngquant_all ${kodak_dir} ${pngquant_kodak}

python2 -m daltonquant.analysis.evaluate\
  ${specimen_database}\
  ${images}\
  230,204,179,153,128\
  ${tiny_png_key}\
  ${tiny_png_folder}\
  --outdir ${compression_one_dir}\
  --userids ${three_users}\
  --alpha 1.0\

python2 -m daltonquant.analysis.compression_hue_accuracies \
  ${specimen_database}\
  ${pngquant_kodak}\
  ${compression_one_dir}\
  "nonlinear transform (nn)"\
  ${user1}\
  --outdir ${analysis_dir}\
  --ncolors 230,204,179,153,128


python2 -m daltonquant.analysis.compression_hue_accuracies\
  ${specimen_database}\
  ${pngquant_kodak}\
  ${compression_one_dir}\
  "linear transform (least-squares)" \
  ${user1} \
  --outdir ${analysis_dir}\
  --ncolors 230,204,179,153,128


# Impact of reduced histories
# computes mean RSS when using smaller %s of the specimen history
# and spearman rho over top 200 color merger candidates
# for user 1 and a sample of kodak images
python2 -m daltonquant.analysis.evaluate_limited_history \
  ${specimen_database} \
  ${three_users} \
  ${images}\
  ${analysis_dir}

python2 -m daltonquant.analysis.evaluate_multi_objective\
  ${specimen_database}\
  ${user1} \
  204\
  ${images}\
  ${analysis_dir}

# Images transformed
# these specific examples assume we are running the replication version
# since it reference specific user ids
if [ ${mode} == 'replication' ]
then
  nl_transform_example="${compression_dir}/specimen_nonlinear transform (nn)_pngquant_userid_506A7DDD727F9ABF6E56D7AED542338025787A46_ncolors_230_kodim01.png"
  l_transform_example="${compression_dir}/specimen_linear transform (least-squares)_pngquant_userid_506A7DDD727F9ABF6E56D7AED542338025787A46_ncolors_230_kodim01.png"

  python2 -m daltonquant.analysis.image_with_transform \
    ${specimen_database} \
    ${user1} \
    "${nl_transform_example},${l_transform_example}" \
    "non-linear,linear"\
    ${analysis_dir}
fi

