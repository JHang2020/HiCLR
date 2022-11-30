#### HiCLR NTU-60 xview ####

# Pretext
python main.py pretrain_hiclr --config config/ntu60/pretext/pretext_hiclr_xview_joint.yaml
python main.py pretrain_hiclr --config config/ntu60/pretext/pretext_hiclr_xview_motion.yaml
python main.py pretrain_hiclr --config config/ntu60/pretext/pretext_hiclr_xview_bone.yaml

# Linear_eval
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_hiclr_xview_joint.yaml
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_hiclr_xview_motion.yaml
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_hiclr_xview_bone.yaml

# Ensemble
python ensemble_ntu_cv.py
