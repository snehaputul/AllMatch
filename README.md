## Improving 3D Semi-supervised Learning by Effectively Utilizing All Unlabelled Data
This is a Pytorch implementation of Improving 3D Semi-supervised Learning by Effectively Utilizing All Unlabelled Data.


### Requirements
python >= 3.7

pytorch >= 1.6

h5py

scikit-learn



### Example training and testing
```shell script
# train
python main.py --exp_name train --perceptange 2 --num_points 1024 --dataset ModelNet40 --batch_size 24 --ema_m 0.99 --unlabeled_ratio 5 --epochs 350 --masking_epoch 50 --lr 7.5e-05 --fake_epoch 5 --u_lambda 1.0 --lambda_ce 1 --unsupcon_lambda 0.2 --supcon_lambda 0 --nl_lambda 1

