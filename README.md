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
