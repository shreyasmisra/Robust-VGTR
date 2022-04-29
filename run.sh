#!/bin/bash

#SAVEDATA='/home/ubuntu/VGTR/store/logs_co_attn_encoder/output/' # for test

#python main.py --gpu 0 --savepath store/logs_co_attn_encoder/resume_17/ --resume store/logs_co_attn_encoder/model/model_refcoco_batch_48/model_refcoco_checkpoint.pth.tar --data_perc 0.33
python main.py --gpu 0 --savepath store/baseline --resume store/baseline/model/model_refcoco_batch_48/model_refcoco_checkpoint.pth.tar --data_perc 0.3
