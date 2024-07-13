# Improving Adversarial Energy-Based Model via Diffusion Process
Pytorch implementation of the paper [Improving Adversarial Energy-Based Model via Diffusion Process](https://arxiv.org/pdf/2403.01666)

by [Cong Geng](https://gengcong940126.github.io), [Tian Han](https://thanacademic.github.io), [Pengtao Jiang](https://pengtaojiang.github.io), Hao Zhang, Jinwei Chen, [Søren Hauberg](https://www2.compute.dtu.dk/~sohau), and [Bo Li](https://libraboli.github.io).
## Requirements
Run the following to install a Python environment:
```
conda env create -f environment.yml
```
This .yml file may be a little bit messy, but it has been verified to work  across different devices.

## Set up datasets
Following [DDGAN](https://arxiv.org/pdf/2112.07804), for large datasets, we store the data in LMDB datasets for I/O efficiency. Check [here](https://github.com/NVlabs/NVAE#set-up-file-paths-and-data) for information regarding dataset preparation.

## Training Denoising Diffusion Adversarial Energy-Based Model (DDAEBM)

```
python train_toy.py
```
We have only released the training code for toy data so far. More implementations for image datasets are still being prepared. Stay tuned.

## Acknowledgements
- [DDGAN](https://github.com/NVlabs/denoising-diffusion-gan)
- [score_sde](https://github.com/yang-song/score_sde_pytorch)

## BibTeX
```
@article{geng2024improving,
  title={Improving Adversarial Energy-Based Model via Diffusion Process},
  author={Geng, Cong and Han, Tian and Jiang, Peng-Tao and Zhang, Hao and Chen, Jinwei and Hauberg, S{\o}ren and Li, Bo},
  journal={International Conference on Machine Learning},
  year={2024}
}
```
