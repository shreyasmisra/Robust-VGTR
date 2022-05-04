#!/bin/bash

#SAVEDATA='/home/ubuntu/VGTR/store/logs_co_attn_encoder/output/' # for test

#python main.py --gpu 0 --savepath "/home/ubuntu/VGTR/store/new_attention/" --resume "/home/ubuntu/VGTR/store/new_attention/model/model_refcoco_batch_48/model_refcoco_checkpoint.pth.tar" --data_perc 0.3
#python main.py --gpu 0 --savepath store/logs_co_attn_encoder/resume_17/ --resume store/logs_co_attn_encoder/resume_17/model/model_refcoco_batch_48/model_refcoco_checkpoint.pth.tar --data_perc 0.3
#python main.py --gpu 0 --savepath store/baseline/resume --resume "/home/ubuntu/VGTR/store/baseline/resume/model/model_refcoco_batch_48/model_refcoco_checkpoint.pth.tar" --data_perc 0.3

python main.py --gpu 0 --test --dataset refcoco+ --pretrain "/home/ubuntu/VGTR/store/new_attention/model/model_refcoco_batch_48/model_refcoco_best.pth.tar" --data_perc 0.99 --save_data "/home/ubuntu/VGTR/store/new_attention/output_refcoco_plus/" --split testB