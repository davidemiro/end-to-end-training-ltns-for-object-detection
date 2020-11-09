# Usage

- `git checkout retinanet`
- activate conda enviroment **rne** (or create with 'retinanet-enviroment.yml') 


Experiment list reference can be found [here](https://docs.google.com/spreadsheets/d/1Zs-IN9gQ8vOcLFdOX4foV-iS1ClwsyGjXc5pwQDMUvU/edit#gid=0)

### List of experiments commands:

##### S0

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-4 --steps 500 --epochs 80 --labdir ./labnotes --experiment S0 csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`

##### S1

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-5 --steps 500 --epochs 80 --labdir ./labnotes --experiment S1 csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`

##### S2

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-6 --steps 500 --epochs 80 --labdir ./labnotes --experiment S2 csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`

##### S3

note: changed anchor sizes in utils/anchors.py to sizes = [32, 64, 128, 256, 512] (line 227)

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-5 --steps 500 --epochs 80 --labdir ./labnotes --experiment S3 csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`


##### S3B-fail

note: Only 3 anchor sizes--> anchor sizes considered only at level 5, 6 and 7 in utils/anchors.py (line 226)

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-5 --steps 500 --epochs 80 --labdir ./labnotes --experiment S3B csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`



##### S4

note: changed focal loss parameter gamma=5 in keras_retinanet/losses.py (line 21)

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-5 --steps 500 --epochs 80 --labdir ./labnotes --experiment S4 csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`



# Usage

- `git checkout retinanet`
- activate conda enviroment **rne** (or create with 'retinanet-enviroment.yml') 


Experiment list reference can be found [here](https://docs.google.com/spreadsheets/d/1Zs-IN9gQ8vOcLFdOX4foV-iS1ClwsyGjXc5pwQDMUvU/edit#gid=0)

### List of experiments commands:

##### S0

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-4 --steps 500 --epochs 80 --labdir ./labnotes --experiment S0 csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`

##### S1

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-5 --steps 500 --epochs 80 --labdir ./labnotes --experiment S1 csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`

##### S2

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-6 --steps 500 --epochs 80 --labdir ./labnotes --experiment S2 csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`

##### S3

note: changed anchor sizes in utils/anchors.py to sizes = [32, 64, 128, 256, 512] (line 227)

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-5 --steps 500 --epochs 80 --labdir ./labnotes --experiment S3 csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`


##### S3B-fail

note: Only 3 anchor sizes--> anchor sizes considered only at level 5, 6 and 7 in utils/anchors.py (line 226)

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-5 --steps 500 --epochs 80 --labdir ./labnotes --experiment S3B csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`



##### S4

note: changed focal loss parameter gamma=5 in keras_retinanet/losses.py (line 21)

`python ./keras_retinanet/bin/train.py --compute-val-loss --lr 1e-5 --steps 500 --epochs 80 --labdir ./labnotes --experiment S4 csv ./calc_mass_csv/train-annotations.csv ./calc_mass_csv/class-mapping.csv --val-annotations ./calc_mass_csv/val-annotations.csv`



