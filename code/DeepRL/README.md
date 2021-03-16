# RPG Algorithm

install pytorch, tensorflow, opencv-python, tensorboardx, gym


To run on the 2-way diabolical combination lock, run:

'''
python -i run.py -alg ppo-rpg -bonus_coeff 5.0 -horizon 3 -env diabcombolockhallway -lr 0.0001 -seed 1 -bonus_type counts
python -i run.py -alg ppo-rpg -bonus_coeff 5.0 -horizon 100 -env MountainCarContinuous-v0 -lr 0.0001 -seed 1 -bonus_type rbf-kernel
'''


