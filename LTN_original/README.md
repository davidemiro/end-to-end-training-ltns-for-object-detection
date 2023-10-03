# LOGIC TENSOR NETWORKS FOR SEMANTIC IMAGE INTERPRETATION

- This repository contains an implementation of Logic Tensor Network for Semantic Image Interpretation, the generated grounded theories, python scripts for baseline and grounded theories evaluation and the PascalPart dataset.
- All the material in the repository is the implementation of the paper *Logic Tensor Networks for Semantic Image Interpretation*.
- Download the repository, unzip the file `LTN_SII.zip` and move into the `LTN_SII/code` folder.
- Before execute LTN install TensorFlow 0.12 library https://www.tensorflow.org/. We tested LTN on Ubuntu Linux with Python 2.7.6.
- You can use/test the trained grounded theories or train a new grounded theory, see how-tos below.

## Structure of LTN_SII folder

- `pascalpart_dataset.tar.gz`: it contains the annotations (e.g., small specific parts are merged into bigger parts) of pascalpart dataset in pascalvoc style. This folder is necessary if you want to train Fast-RCNN (https://github.com/rbgirshick/fast-rcnn) on this dataset for computing the grounding/features vector of each bounding box.
    - `Annotations`: the annotations in `.xml` format. To see bounding boxes in the images use the pascalvoc devkit http://host.robots.ox.ac.uk/pascal/VOC/index.html.
    - `ImageSets`: the split of the dataset into train and test set according to every unary predicate/class. For further information See pascalvoc format at devkit http://host.robots.ox.ac.uk/pascal/VOC/index.html.
    - `JPEGImages`: this folder is currently empty but you can download the original images from http://host.robots.ox.ac.uk/pascal/VOC/voc2010/.

- `code`: it contains the data, the output folder and the source code of LTN.
    - `data`: the training set, the test set and the ontology that defines the mereological axioms.
    - `results`: the output of the evaluation of the baseline and of the grounded theories;
    - `models`: the trained grounded theories.

## How to train a grounded theory

```sh
$ python train.py
```
- Trained grounded theories are in the `models` folder.

## How to evaluate the grounded theories and the baselines

```sh
$ python evaluate.py
```
- Results are in the `results` folder.
- More detailed results are in `results/report.csv`.
