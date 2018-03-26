## DaltonQuant
##### Compression for color vision deficient individuals
--------------

## Summary
`DaltonQuant` is a research prototype image compression tool that
uses data collected through the iOS game [Specimen](https://itunes.apple.com/us/app/specimen-a-game-about-color/id999930535?mt=8)
to improve compression for colorblind users. The tool queries a
database constructed from Specimen data, extracts relevant records
for the user identifier provided (or extracts a set of identifiers
associated with potentially color blind users), uses these records
to construct two simple transformations, and applies these to an
image that has already been compressed with an off-the-shelf
compressor.

This document aims to provide instructions to reproduce all
data/figures used in the associated research paper.


## Installation
`DaltonQuant` is implemented in Python 2.7. We ran all experiments
using 2.7.12. We assume you have standard software installed, such
as `git`.

You may find it convenient to use Python's `virtualenv` to create
a separate environment for package installation.


```
virtualenv -p python2.7 env/
source env/bin/activate
```

You can install the necessary packages and (almost all) tools
necessary by cloning this repository, and following the commands
below


```
git clone git@github.com:josepablocam/daltonquant.git daltonquant/
cd daltonquant/
make
```

Some parts of the installation process are not yet automated (and
some cannot be automated, as they require user registration) Please
see the following section for these steps.



## (Necessary) manual installation steps
After you have followed the steps in the prior section, please make
sure to complete the list below.

* Request access to the Specimen data by emailing jcamsan@mit.edu
* Build the Specimen database (the necessary sources are downloaded in the prior step)
  ```
  cd specimen-tools/
  specimen-tools/scripts/build_db.py <path-to-specimen-data> <database-file>
  ```
  See [`specimen-tools`](https://github.com/josepablocam/specimen-tools) for more details.
* Register for `tinypng` usage. You will need to sign up at their 
  [website](https://tinypng.com/). You should receive a link
  so that you can sign up for a developer API key. We'll provide
  this to `DaltonQuant` by modifying the corresponding values in a bash script.

  

### Reproducing Results
You should update your TinyPNG information in the `reproduce.sh` script.
In particular, change the value of `tiny_png_key` to correspond to your png key.
The current value is set to empty and will fail if executed.

Reproducing the results should be as simple as

```
./reproduce.sh
```

This will create multiple directories under a single directory named `generated`.
The directories of interest are:
  
  * `analysis_results`: Contains figures and csv files for the analysis performed on
    any compressed images.
  * `compression_results`: Contains a large collection of compressed images for
  different parameters. This also contains a summary of the file sizes for each image and compression.
  * `compression_results_alpha_1`: Contains a collection of compressed images with
  the multi-objective optimization weight (alpha) set to 1.0.
  * `pcvd_users`: Contains figures and csv files for the potentially color vision deficient
  users identified and used for experiments.


Note that `reproduce.sh` assumes the specimen database has been built in the
top-level `daltonquant` directory and is named `specimendb`. You are of course
free to modify that but just make sure to change the correct path in the
bash script itself then.

### Caveats
The source code and directory structure in this project is under active work.

This document will be maintained to reflect the appropriate changes
in commands for reproduction.
