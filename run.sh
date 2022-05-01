<<<<<<< HEAD
#!/bin/bash

#SAVEDATA='/home/ubuntu/VGTR/store/logs_co_attn_encoder/output/' # for test

python main.py --gpu 0 --savepath store/logs_co_attn_encoder/ --dataset refcoco+ 
#python main.py --gpu 0 --savepath store/logs_co_attn_encoder/resume_26/ --resume store/logs_co_attn_encoder/resume_17/model/model_refcoco_batch_48/model_refcoco_checkpoint.pth.tar --data_perc 0.45
#python main.py --gpu 0 --savepath store/baseline --resume store/baseline/model/model_refcoco_batch_48/model_refcoco_checkpoint.pth.tar --data_perc 0.3

#python main.py --gpu 0 --test --dataset refcoco+ --pretrain store/logs_co_attn_encoder/resume_17/model/model_refcoco_batch_48/model_refcoco_best.pth.tar --data_perc 0.99 --save_data store/logs_co_attn_encoder/output_refcoco_plus/ --split testA
=======
#python main.py --gpu 0 --savepath store/contrastive_loss --data_perc 0.33 
python main.py --gpu 0 --savepath store/contrastive_loss --resume store/contrastive_loss/model/model_refcoco_batch_48/model_refcoco_checkpoint.pth.tar --data_perc 0.33
>>>>>>> 25fd0c973e7a01e6dc744daaecb4a4bee05b09fc
