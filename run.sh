#!/bin/bash

#SAVEDATA='/home/ubuntu/VGTR/store/logs_co_attn_encoder/output/' # for test

#python main.py --gpu 0 --savepath store/logs_co_attn_encoder/ --dataset refcoco+ 
#python main.py --gpu 0 --savepath store/logs_co_attn_encoder/resume_26/ --resume store/logs_co_attn_encoder/resume_17/model/model_refcoco_batch_48/model_refcoco_checkpoint.pth.tar --data_perc 0.45
#python main.py --gpu 0 --savepath store/baseline/resume --resume store/baseline/model/model_refcoco_batch_48/model_refcoco_checkpoint.pth.tar --data_perc 0.3

python main.py --gpu 0 --test --dataset refcoco --pretrain "/home/ubuntu/VGTR/store/logs_co_attn_encoder/resume_17/model/model_refcoco_batch_48/model_refcoco_best.pth.tar" --data_perc 0.99 --save_data "/home/ubuntu/VGTR/store/logs_co_attn_encoder/output/" --split testA