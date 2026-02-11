# Code of PhaSpa
# Create the environment
conda env create -f env.yaml

# Activate the environment
conda activate phaspa
# Basic train
python run.py --valid --dataset sevir --epochs 100 --batch_size 4
## Acknowledgement

We refer to implementations of the following repositories and sincerely thank their contribution for the community:
- [AlphaPre](https://github.com/linkenghong/AlphaPre?tab=readme-ov-file)
- [OpenSTL](https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/README.md)
- [DiffCast](https://github.com/DeminYu98/DiffCast)
