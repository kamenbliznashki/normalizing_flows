# Normalizing flows

Reimplementations of density estimation algorithms from:
* [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057)
* [Density Estimation using RealNVP](https://arxiv.org/abs/1605.08803)
* [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770)

## Masked Autoregressive Flow
https://arxiv.org/abs/1705.07057

Reimplementation of MADE, MAF, Mixture of Gaussians MADE, Mixture of
Gausssians MAF, and RealNVP modules on UCI datasets and MNIST.

#### Results
Average test log likelihood for un/conditional density estimation (cf.
Table 1 & 2 in paper):

| Model | POWER | GAS | HEPMASS | MINIBOONE | BSDS300 | MNIST (uncond) | MNIST (cond) |
| --- | --- | --- |
| MADE | -3.10 +/- 0.02 | 2.53 +/- 0.02 | -21.13 +/- 0.01 | -15.36 +/- 15.06 | 146.42 +/- 0.14 | -1393.67 +/- 1.90 | -1338.12 +/- 1.75 |
| MADE MOG | 0.37 +/- 0.01 | 8.08 +/- 0.02 | -15.70 +/- 0.02 | -11.64 +/- 0.44 | 153.56 +/- 0.28 | -1023.13 +/- 1.69 | -1010.51 +/- 1.64 |
| RealNVP (5) | -0.49 +/- 0.01 | 7.01 +/- 0.06 | -19.96 +/- 0.02 | -16.88 +/- 0.21 | 148.34 +/- 0.26 | -1279.76 +/- 9.91 | -1275.06 +/- 14.23 |
| MAF (5) | 0.03 +/- 0.01 | 6.23 +/- 0.01 | -17.97 +/- 0.01 | -11.57 +/- 0.21 | 153.53 +/- 0.27 | -1272.70 +/- 1.87 | -1270.12 +/- 6.89 |
| MAF MOG (5) | 0.09 +/- 0.01 | 7.96 +/- 0.02 | -17.29 +/- 0.02 | -11.27 +/- 0.41 | 153.35 +/- 0.26 | -1080.46 +/- 1.53 | -1067.12 +/- 1.54 |

Toy density model (cf. Figure 1 in paper):

| Target density | Learned density with MADE <br> and random numbers driving MADE | Learned density with MAF 5 layers <br> and random numbers driving MAF |
| --- | --- | --- | --- | --- |
| ![fig1a](images/maf/figure_1a.png) | ![fig1b](images/maf/figure_1b.png) | ![fig1c](images/maf/figure_1c.png) |

Class-conditional generated images from MNIST using MAF (5) model; generated data arrange by decreasing log probability (cf. Figure 3 in paper):

![mafmnist](images/maf/generated_samples_maf5.png)

#### Usage
To train model:
```
python maf.py -- train \
              -- model=['made' | 'mademog' | 'maf' | 'mafmog' | 'realnvp'] \
              -- dataset=['POWER' | 'GAS' | 'HEPMASS' | 'MINIBOONE' | 'BSDS300' | MNIST'] \
              -- n_blocks=[for maf/mafmog and realnvp specify # of MADE-blocks / coupling layers] \
              -- n_components=[if mixture of Gaussians, specify # of components] \
              -- conditional [if MNIST, can train class-conditional log likelihood] \
              -- [add'l options see py file]
```

To evaluate model:
```
python maf.py -- evaluate \
              -- restore_file=[path to .pt checkpoint]
              -- [options of the saved model: n_blocks, n_hidden, hidden_size, n_components, conditional]
```

To generate data from a trained model (for MNIST dataset):
```
python maf.py -- generate \
              -- restore_file=[path to .pt checkpoint]
              -- dataset='MNIST'
              -- [options of the saved model: n_blocks, n_hidden, hidden_size, n_components, conditional]
```

#### Datasets

Datasets and preprocessing code are forked from the MAF authors' implementation [here](https://github.com/gpapamak/maf#how-to-get-the-datasets). The unzipped datasets should be symlinked into the `./data` folder or the data_dir argument should be specified to point to the actual data.

#### References
* The original Theano implementation by the authors https://github.com/gpapamak/maf/
* https://github.com/ikostrikov/pytorch-flows


## Variational inference with normalizing flows
Implementation of [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770)

#### Results
Density estimation of 2-d test energy potentials (cf. Table 1 & Figure 3 in paper).

| Target density | Flow K = 2 | Flow K = 32 | Training parameters |
| --- | --- | --- | --- |
| ![uz1](images/nf/nf_uz1_target_potential_density.png) | ![uz1k2](images/nf/nf_uz1_flow_k2_density.png) | ![uz1k32](images/nf/nf_uz1_flow_k32_density.png) | weight init Normal(0,1), base dist. scale 2 |
| ![uz2](images/nf/nf_uz2_target_potential_density.png) | ![uz2k2](images/nf/nf_uz2_flow_k2_density.png) | ![uz2k32](images/nf/nf_uz2_flow_k32_density.png) | weight init Normal(0,1), base dist. scale 1 |
| ![uz3](images/nf/nf_uz3_target_potential_density.png) | ![uz3k2](images/nf/nf_uz3_flow_k2_density.png) | ![uz3k32](images/nf/nf_uz3_flow_k32_density.png) | weight init Normal(0,1), base dist. scale 1, weight decay 1e-3 |
| ![uz4](images/nf/nf_uz4_target_potential_density.png) | ![uz4k2](images/nf/nf_uz4_flow_k2_density.png) | ![uz4k32](images/nf/nf_uz4_flow_k32_density.png) | weight init Normal(0,1), base dist. scale 4, weight decay 1e-3 |


#### Usage
To train model:
```
python planar_flow.py -- train \
                      -- target_potential=[choice from u_z1 | u_z2 | u_z3 | u_z4] \
                      -- flow_length=[# of layers in flow] \
                      -- [add'l options]
```
Additional options are: base distribution (q0) scale, weight initialization
scale, weight decay, learnable first affine layer (I did not find adding an affine layer beneficial).

To evaluate model:
```
python planar_flow.py -- evaluate \
                      -- restore_file=[path to .pt checkpoint]
```

#### Useful resources
* https://github.com/casperkaae/parmesan/issues/22


## Dependencies
* python 3.6
* pytorch 0.4
* numpy
* matplotlib
