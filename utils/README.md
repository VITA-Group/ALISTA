# ALISTA: Analytic Weights Are As Good As Learned Weights in LISTA
This repository is for Analytic-LISTA networks proposed in the following paper:

[Jialin Liu\*](http://www.math.ucla.edu/~liujl11/pub.html),
[Xiaohan Chen\*](http://people.tamu.edu/~chernxh),
[Zhangyang Wang](http://www.atlaswang.com/) and
[Wotao Yin](http://www.math.ucla.edu/~wotaoyin/)
"ALISTA: Analytic Weights Are As Good As Learned Weights in LISTA", accepted at
ICLR 2019. The pdf can be found [here](https://openreview.net/pdf?id=B1lnzn0ctQ).

\*: These authors contributed equally and are listed alphabetically.

The code is tested in Linux environment (Python: 3.5.2, Tensorflow: 1.12.0,
CUDA9.0) with Nvidia GTX 1080Ti GPU.


<!-- vim-markdown-toc GFM -->

* [Introduction](#introduction)
* [Run the codes](#run-the-codes)
    * [Generate problem files](#generate-problem-files)
    * [Baseline: LISTA-CPSS](#baseline-lista-cpss)
    * [TiLISTA](#tilista)
    * [ALISTA](#alista)
        * [Solve analytic weight from measurement matrix](#solve-analytic-weight-from-measurement-matrix)
        * [ALISTA with analytic weight](#alista-with-analytic-weight)
    * [Robust ALISTA](#robust-alista)
        * [Pre-train encoders](#pre-train-encoders)
        * [Jointly train encoder and decoder](#jointly-train-encoder-and-decoder)
        * [Testing](#testing)
    * [Data augmented decoders](#data-augmented-decoders)
    * [Convolutional LISTA for natural image denoising](#convolutional-lista-for-natural-image-denoising)
* [Cite this work](#cite-this-work)

<!-- vim-markdown-toc -->

## Introduction
Deep neural networks based on unfolding an iterative algorithm, for example,
LISTA (learned iterative shrinkage thresholding algorithm), have been an
empirical success for sparse signal recovery. The weights of these neural
networks are currently determined by data-driven “black-box” training. In this
work, we propose Analytic LISTA (ALISTA), where the weight matrix in LISTA is
computed as the solution to a data-free optimization problem, leaving only the
stepsize and threshold parameters to data-driven learning. This signiﬁcantly
simpliﬁes the training. Speciﬁcally, the data-free optimization problem is based
on coherence minimization. We show our ALISTA retains the optimal linear
convergence proved in (Chen et al., 2018) and has a performance comparable to
LISTA. Furthermore, we extend ALISTA to convolutional linear operators, again
determined in a data-free manner. We also propose a feed-forward framework that
combines the data-free optimization and ALISTA networks from end to end, one
that can be jointly trained to gain robustness to small perturbations in the
encoding model.

## Run the codes

### Generate problem files
To run most of experiments in this repository, you need to first generate an
instance of `Problem` or `ProblemConv` class, which you can find in
`utils/prob.py` or `utils/prob_conv.py` file.

Run the following command to generate a random measurement matrix:

```
python3 utils/prob.py --M 250 --N 500 \
    --pnz 0.1 --SNR inf --con_num 0.0 --column_normalized True
```

Explation for the options:
* `--M`: the dimension of measurements.
* `--N`: the dimension of sparse signals.
* `--pnz`: the approximate of non-zero elements in sparse signals.
* `--SNR`: the *signal-to-noise* ratio in dB unit in the measurements. `inf`
  means noiseless setting.
* `--con_num`: the condition number. 0.0 (default) means the condition number
  will not be changed.
* `--column_normalized`: whether normalize the columns of the measurement matrix
  to unit l-2 norm.

The generated will be saved to the `experiments/m250_n500_k0.0_p0.1_s40/prob.npz`.
If you want to generate a problem from an existing measurement matrix, which
should be saved in Numpy `npy` file format, use `--load_A` option with the path
to the matrix file. In this case, options `--M` and `--N` will be overwriiten by
the shape of loaded matrix.

### Baseline: LISTA-CPSS
Use the baseline model *LISTA-CPSS* in [this paper](http://papers.nips.cc/paper/8120-theoretical-linear-convergence-of-unfolded-ista-and-its-practical-weights-and-thresholds)
to basicly explain how to train and test models. To train or test a LISTA-CPSS
model, use the following command:
```
python3 main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
    --net LISTA_cpss -T 16 -p 1.2 -maxp 13 \
    --scope LISTA_cpss --exp_id 0
```

Explanation for the options (all optinos are parsed in `config.py`):
* `--task_type`: the task on which you will train/test your model. Possible
  values are:
  * `sc` standing for normal simulated sparse coding algorithm;
  * `cs` for natural image compressive sensing;
  * `denoise` for natural image denoising using convolutional LISTA;
  * `encoder` for encoder pre-training; and
  * `robust` for robustness training.
* `-g/--gpu`: the id of GPU used. GPU 0 will be used by default.
* `-t/--test` option indicates training or testing mode. Use this option for
  testing.
* `-n/--net`: specifies the network to use.
* `-T`: the number of layers.
* `-p/--percent`: the percentage of entries to be added to the support in each
  layer.
* `-maxp/--max_percent`: maximum percentage of entries to be selected.
* `--scope`: the name of variable scope of model variables in TensorFlow.
* `--exp_id`: experiment id, used to differentiate experiments with the same
  setting.

### TiLISTA

To train or test a TiLISTA (Tied-LISTA) network, run the following command:
```
python3 main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
    --net TiLISTA -T 16 -p 1.2 -maxp 13 \
    --scope TiLISTA --exp_id 0
```

### ALISTA
#### Solve analytic weight from measurement matrix
Use MatLab script `matlabs/CalculateW.m` to solve an analytic weight matrix from
an existing measurement matrix, which should be saved as a MatLab `mat` file
with key word `D` for the matrix. We provide an example in `data/D.mat`, which
is the same matrix as in `experiments/m250_n500_k0.0_p0.1_sinf/prob.npz`. Use
`matlabs/CalculateW_conv.m` for convolutional dictionaries. We provide a
pre-solved weight saved as `data/W.npy`.

#### ALISTA with analytic weight
```
python3 main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
    --net ALISTA -T 16 -p 1.2 -maxp 13 -W ./data/W.npy \
    --better_wait 2000 \
    --scope ALISTA --exp_id 0
```

Explanation for options:
* `-W`: path to the specified weight matrix.
* `--better_wait`: maximum waiting time for a better validation accuracy before
  going to the next training stage. ALISTA model has 2T parameters, thus having
  a very stabilized training process. Therefore, we can use a smaller waiting
  time than LISTA-CPSS (use `--better_wait 5000` by default).

### Robust ALISTA
To train a robust ALISTA model, you need 3 steps:
1. Pre-train a encoder.
2. Pre-train a ALISTA decoder. We can use the ALISTA modeled trained in the above
   section.
3. Jointly train the encoder and the decoder.
#### Pre-train encoders
```
python3 main.py --task_type encoder -g 0 \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
    --net AtoW_grad --eT 4 --Binit default --eta 1e-3 --loss rel2 \
    --Abs 16 --encoder_psigma 1e-2 --encoder_pre_lr 1e-4 \
    --scope AtoW --exp_id 0
```

Explanation for the options:
* `--net AtoW_grad`: the encoding model unfoled from projected gradient descent.
* `--eT`: the number of layers in the encoder.
* `--Binit`: use the default method to initialize weights in the encoder. You
  can use random initialization by specifying `normal` or `uniform` here.
* `--eta`: the initial step size in the projected gradient descent.
* `--loss`: the objective function in the original optimization, and the cost
  function used to train the encoder. `rel2` means *reweighted l2*.
* `--Abs`: the batch size to sample perturbed matrices.
* `--encoder_psigma`: the noise level to perturb the measurement matrix.
* `--encoder_pre_lr`: the initial learning rate for pre-training the encoder.

#### Jointly train encoder and decoder

```
python3 main.py --task_type robust -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
    --net robust_ALISTA \
    --eT 4 --Binit default --eta 1e-3 --loss rel2 --encoder_scope AtoW \
    --encoder_psigma 1e-2 --encoder_pre_lr 1e-4 --encoder_id 0 \
    --dT 16 --lam 0.4 -p 1.2 -maxp 13 -W .data/W.npy \
    --decoder_scope ALISTA --decoder_id 0 \
    --psigma_max 2e-2 --psteps 5 --msigma 0.0 \
    --encoder_lr 1e-9 --decoder_lr 1e-4 \
    --Abs 4 --xbs 16 --maxit 50000 --exp_id 0
```

Explanation for the options:
* `--dT`: the number of layers in the decoder.
* `--psigma_max`: the maximum level of perturbations during the joint training.
* `--psteps`: the number of steps of the curriculum training where we gradually
  increase the level of perturbations till `psigma_max`.
* `--msigma`: the level of measurement noises during the joint training.
* `--xbs`: the batch size used to generate sparse signals for each perturbed
  measurement matrix. The total number of measurement-signal pairs in a batch is
  `Abs` times `xbs`.
* `--maxit`: the maximum number of training steps for each curriculum training
  stage in the whole training process.

#### Testing
Use the above command with `-t/--t` option to test. For testing we genereate a
sample of perturbed measurement matrices, which you can download using this
[Dropbox link](https://www.dropbox.com/s/9m7s1g1u4apy4wx/robust_test_A.npz?dl=0).
The original measurement matrix used to generate this file is the same as in
`experiments/m250_n500_k0.0_p0.1_sinf/prob.npz`.

### Data augmented decoders
To train or test a data-augmented decoding model, run the following command (use
TiLISTA for example):
```
python3 main.py --task_type robust -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 --column_normalized True \
    --net TiLISTA -T 16 --lam 0.4 -p 1.2 -maxp 13 \
    --psigma_max 2e-2 --psteps 5 --msigma 0.0 \
    --decoder_lr 1e-4 --Abs 4 --xbs 16 --maxit 50000 \
    --scope TiLISTA --exp_id 0
```

### Convolutional LISTA for natural image denoising
1. Download BSD500 dataset. Split into train, validation and test sets as you
   wish.
2. Genereate the tfrecords using:
   ```
   python3 utils/data.py --task_type denoise \
       --dataset_dir /path/to/your/[train,val,test]/folder \
       --out_dir path/to/the/folder/to/store/tfrecords \
       --out_file [train,val,test].tfrecords \
       --suffix jpg
   ```
3. Learn a convolutional dictionary from the BSD500 dataset using the algorithm
   in the paper [First- and Second-Order Methods for Online Convolutional
   Dictionary Learning](https://arxiv.org/abs/1709.00106). Or use the dictionary
   proveided in `data/D3_M100_lam0.05.mat`.
4. Generate a problem file using the learned dictionary and the following
   command:
   ```
   python3 utils/prob_conv.py --conv_d 3 --conv_m 100 --conv_lam 0.05 \
       --load_dict ./data/D3_M100_lam0.05.mat
   ```
   where `--conv_d` is the size of filters in the dictionary, `--conv_m` is the
   number of filters, `--conv_lam` is the parameter used in convolutional
   dictionary learning algorithm, and `--load_dict` specifies the dictionary to
   be loaded and saved. The generated problem file will be saved to
   `experiments/denoise_d3_m100_lam0.05/prob.npz`.
5. Train and test the convolutional denoising model (use Conv-TiLISTA as an
   example):
   ```
   python3 main.py --task_type denoise -g 0 [-t] \
       --net TiLISTA -T 5 --lam 0.1 --conv_alpha 0.1 \
       --sigma 20 --height_crop 321 --width_crop 321 \
       --num_epochs -1 --tbs 4 --vbs 16 \
       --data_folder data/denoise_tfrecords
       --train_file training_tfrecords_filename \
       --val_file validation_tfrecords_filename
   ```
   Explanation for the options:
   * `--conv_alpha`: the initial step size in learned convolutional model.
   * `--sigma`: the noise level in the images.
   * `--height_crop` and `--width_crop`: size of cropped images in training.
   * `--num_epochs`: the number of epochs to train over the BSD500 training set.
     The default `-1` value means infinite nubmer of opochs. The training will
     be ended as in `sc` task.
   * `--tbs` and `--vbs`: training and validation batch sizes.
   * `--data_folder`: the path to the folder that holds the tfrecords files.

## Cite this work
If you find our code helpful in your resarch or work, please cite our paper.
```
@inproceedings{
liu2018alista,
title={{ALISTA}: Analytic Weights Are As Good As Learned Weights in {LISTA}},
author={Jialin Liu and Xiaohan Chen and Zhangyang Wang and Wotao Yin},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=B1lnzn0ctQ},
}
```
