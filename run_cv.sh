#### HiCLR NTU-60 xview ####

# Pretext
python main.py pretrain_hiclr --config config/release/ntu60/pretext/pretext_hiclr_xview_joint.yaml
python main.py pretrain_hiclr --config config/release/ntu60/pretext/pretext_hiclr_xview_motion.yaml
python main.py pretrain_hiclr --config config/release/ntu60/pretext/pretext_hiclr_xview_bone.yaml

# Linear_eval
python main.py linear_evaluation --config config/release/ntu60/linear_eval/linear_eval_hiclr_xview_joint.yaml
python main.py linear_evaluation --config config/release/ntu60/linear_eval/linear_eval_hiclr_xview_motion.yaml
python main.py linear_evaluation --config config/release/ntu60/linear_eval/linear_eval_hiclr_xview_bone.yaml

#finetune
python main.py finetune_evaluation --config config/release/gcn_ntu60/finetune/xview_joint.yaml
# Ensemble
python ensemble_ntu_cv.py

#visualize
#python main.py vis_evaluation --config config/release/ntu60/finetune/xview_motion.yaml