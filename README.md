# ReSSL

## How to run
Train model in SSL manner
```bash
python ressl.py
```
After training the final checkpoint is saved in `checkpoints/ressl-{dataset}.pth`.

## How to validate

Linear head training and validation of trained checkpoint.
```
python linear_eval.py --checkpoint ressl-{dataset}.pth
```
## Evaluating on downstream tasks using EVAR
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

1. Setup EVAR library https://github.com/nttcslab/eval-audio-repr#2-setup
2. Prepare dataset and metadata
3. Prepare `evar/config/ressl.yaml`

```bash
cd evar
python lineareval.py config/ressl.yaml <dataset_name>
```
