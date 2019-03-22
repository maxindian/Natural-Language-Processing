# Extending a Parser to Distance Domains

The repository contains the code used to generate the results described in [Extending a Parser to Distant Domains Using a Few Dozen Partially Annotated Examples](http://arxiv.org/abs/1805.06556) from ACL 2018.

## Step
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run 'make' inside the 'EVALB/' directory to compile and 'evalb' executable. This will be called from Python for evaluation.
* Pre-trained models. Before starting, run `unzip models/model_dev=94.48.zip` and `unzip zipped/no_elmo_model_dev=92.34.zip` in the `models/` directory to extract the pre-trained models.


## Experiment

### Training
The model can be trained using the command 'python3 main.py'
