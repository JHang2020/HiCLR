#### HiCLR NTU-60 xsub ####

# Pretext
python main.py pretrain_hiclr --config config/release/ntu60/pretext/pretext_hiclr_xsub_joint.yaml
python main.py pretrain_hiclr --config config/release/ntu60/pretext/pretext_hiclr_xsub_motion.yaml
python main.py pretrain_hiclr --config config/release/ntu60/pretext/pretext_hiclr_xsub_bone.yaml

# Linear_eval
python main.py linear_evaluation --config config/release/ntu60/linear_eval/linear_eval_hiclr_xsub_joint.yaml
python main.py linear_evaluation --config config/release/ntu60/linear_eval/linear_eval_hiclr_xsub_motion.yaml
python main.py linear_evaluation --config config/release/ntu60/linear_eval/linear_eval_hiclr_xsub_bone.yaml

#semi or finetune
python main.py finetune_evaluation --config config/release/gcn_ntu60/finetune/xsub_joint.yaml
# Ensemble
python ensemble_ntu_cs.py

#visualize
#python main.py vis_evaluation --config config/release/ntu60/finetune/xsub_motion.yaml