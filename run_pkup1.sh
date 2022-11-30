#### HiCLR PKU-MMD Part1 xsub ####

# Pretext
python main.py pretrain_hiclr --config config/release/pkummd/pretext/pretext_hiclr_pkup1cs_joint.yaml
python main.py pretrain_hiclr --config config/release/pkummd/pretext/pretext_hiclr_pkup1cs_motion.yaml
python main.py pretrain_hiclr --config config/release/pkummd/pretext/pretext_hiclr_pkup1cs_bone.yaml

# Linear_eval
python main.py linear_evaluation --config config/release/pkummd/linear_eval/linear_eval_hiclr_pkup1cs_joint.yaml
python main.py linear_evaluation --config config/release/pkummd/linear_eval/linear_eval_hiclr_pkup1cs_motion.yaml
python main.py linear_evaluation --config config/release/pkummd/linear_eval/linear_eval_hiclr_pkup1cs_bone.yaml

# Ensemble
python ensemble_pkup1cs.py