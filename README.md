# Teamformer: Scalable Heterogeneous Multi-Robot Team Formation
N. Boehme and Geoffrey Hollinger.Update "Teamformer: Scalable Heterogeneous Multi-Robot Team Formation" 2026 IEEE International Conference on Robotics and Automation (ICRA), 2026

## Steps to install
1) Install python 3.10
2) git clone --recurse-submodules <URL>
3) Create and activate a virtual env
4) Install packages from requirements.txt
5) cd into stable-baselines3 and run "pip install -e ."
6) cd into SuperSuit and run "pip install -e ."

## To Train a Model
1) Update train_config.yaml as desired
2) Run "python train.py"

## To Test a Model
1) Add trained model paths to test_config.yaml
2) Run "python test.py"