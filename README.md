# ReSSL-A

Adaptation of ReSSL method to audio.

## How to install

1. Clone repository

```bash
git clone https://github.com/cywinski/ressl_audio.git
cd ressl_audio
```

2. Prepare Conda environment

```bash
conda create -n ressl_env python=3.9
conda activate ressl_env
pip install -r requirements.txt
```

## How to train

### Prepare train dataset

Download the subset of AudioSet from [Kaggle](https://www.kaggle.com/datasets/zfturbo/audioset/data).

### Log to WandB

```bash
wandb login
```

### Run training

To run single training run:

```bash
python ressl.py --audio_dir <path/to/train>
```

To run Optuna hyperparam search run:

```bash
python ressl_optuna.py --audio_dir <path/to/train>
```

The training checkpoints are saved under `checkpoints` directory.

## How to evaluate on downstream tasks using EVAR

Run following to clone your copy

```
echo "Setup EVAR (nttcslab/eval-audio-repr)."
git clone https://github.com/nttcslab/eval-audio-repr.git evar
cd evar
curl https://raw.githubusercontent.com/daisukelab/general-learning/master/MLP/torch_mlp_clf2.py -o evar/utils/torch_mlp_clf2.py
curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/sampler.py -o evar/sampler.py
curl https://raw.githubusercontent.com/daisukelab/sound-clf-pytorch/master/for_evar/cnn14_decoupled.py -o evar/cnn14_decoupled.py

echo "Adding RESSL extensions to the EVAR folder."
sed -i 's/import evar.ar_ressl/import evar.ar_ressl, evar.ar_ressl/' lineareval.py
ln -s ../../ressl external
ln -s ../../to_evar/ar_ressl.py evar
ln -s ../../to_evar/ressl.yaml config
```

1. Setup EVAR library <https://github.com/nttcslab/eval-audio-repr#2-setup>
2. Prepare dataset and metadata
3. Prepare `evar/config/ressl.yaml`

To run validation:

```bash
cd evar
python lineareval.py config/ressl.yaml <dataset_name>
```
