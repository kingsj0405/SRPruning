# VDSR

## Use commands

```bash
# train
python main.py train
CUDA_VISIBLE_DEVICES=0 python main.py train
# Delete configurations
rm -rf ../../REWIND-vdsr-scratch/
rm -rf ../../REWIND-vdsr-scratch/*/v1*
rm -rf ../../REWIND-vdsr-scratch/*/v1.0*
# Format
autopep8 --in-place --aggressive *.py
```