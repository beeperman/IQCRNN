# Recurrent Neural Network Controllers Synthesis with Stability Guarantees for Partially Observed Systems

This repository is an official implementation of the paper: Recurrent Neural Network Controllers Synthesis with Stability Guarantees for Partially Observed Systems (to appear on AAAI 2022). Please consider citing the paper if you find the paper or the code useful for you.

```
@article{gu2021recurrent,
  title={Recurrent Neural Network Controllers Synthesis with Stability Guarantees for Partially Observed Systems},
  author={Gu, Fangda and Yin, He and Ghaoui, Laurent El and Arcak, Murat and Seiler, Peter and Jin, Ming},
  journal={arXiv preprint arXiv:2109.03861},
  year={2021}
}
```

## Requirements

It is recommended that the following packages are installed through pip in the listed order except for python (which you could install with conda in a virtual environment).

* python==3.7.4
* numpy==1.16.4
* scipy==1.3.1
* jupyter
* tensorflow==1.15.0
* gym==0.18.0
* cvxpy==1.0.25 (using `pip install --no-binary ":all:" cvxpy==1.0.25`)
* Mosek==9.2.38 (See mosek.com for license)
* matplotlib==3.1.1

The code is tested on Linux distribution Debian Stretch.

## Training

You may run the training of RNN with or without projection using something like (for example cartpole)

`python3 train_RNN_cartpole.py -rtg --nn_baseline`

or

`python3 train_RNN_cartpole_proj_tilde.py -rtg --nn_baseline`

## Evaluation

See ipython notebooks under `plots/`. The results are reported in the notebooks. 

Pretrained models are included under `data/`.

Note: If you want to run the trajectory notebooks, please run the regular plot (policy gradient) first and then the tilde plot (projected policy gradient). You have to restart the notebook after the regular plot and before the tilde plot.