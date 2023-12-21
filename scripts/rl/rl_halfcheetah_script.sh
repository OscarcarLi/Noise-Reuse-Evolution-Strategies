# FullES
seed=83; N=6; lr="3e-5"
seed=89; N=6; lr="3e-5"
seed=97; N=6; lr="3e-5"
seed=101; N=6; lr="3e-5"
seed=103; N=6; lr="3e-5"
  CUDA_VISIBLE_DEVICES="" python3 main_training.py --application rl --horizon_length 1000 --gym_env_name "HalfCheetah-v4" --rl_init_type normal --rl_init_scale 0.03 --gradient_estimator FullES --num_particles $N --num_gradient_estimators 1 --sigma 0.004 --noloss_normalize --outer_optimizer_specs "SGD||CONSTANT$lr" --outer_iterations 1000 --init_seed 1 --remaining_seed $seed --run_test_during_training --eval_freq 4

# TES
seed=83; W=20; N=300; lr="3e-5"
seed=89; W=20; N=300; lr="3e-5"
seed=97; W=20; N=300; lr="3e-5"
seed=101; W=20; N=300; lr="3e-5"
seed=103; W=20; N=300; lr="3e-5"
seed=83; W=100; N=100; lr="3e-5"
  CUDA_VISIBLE_DEVICES="" python3 main_training.py --application rl --horizon_length 1000 --gym_env_name "HalfCheetah-v4" --rl_init_type normal --rl_init_scale 0.03 --gradient_estimator TruncatedESBiased --num_particles $N --num_gradient_estimators 1 --trunc_length $W --sigma 0.004 --noloss_normalize --outer_optimizer_specs "SGD||CONSTANT$lr" --outer_iterations 1000 --init_seed 1 --remaining_seed $seed --run_test_during_training --eval_freq 4


# PES
seed=83; W=20; N=300; lr="3e-6"
seed=89; W=20; N=300; lr="3e-6"
seed=97; W=20; N=300; lr="3e-6"
seed=101; W=20; N=300; lr="3e-6"
seed=103; W=20; N=300; lr="3e-6"
  CUDA_VISIBLE_DEVICES="" python3 main_training.py --application rl --horizon_length 1000 --gym_env_name "HalfCheetah-v4" --rl_init_type normal --rl_init_scale 0.03 --gradient_estimator TruncatedPESK --num_particles $N --num_gradient_estimators 1 --trunc_length $W --K $W --sigma 0.004 --noloss_normalize --outer_optimizer_specs "SGD||CONSTANT$lr" --outer_iterations 30000 --init_seed 1 --remaining_seed $seed --run_test_during_training --eval_freq 4

# NRES
seed=83; W=20; N=300; lr="3e-5"
seed=89; W=20; N=300; lr="3e-5"
seed=97; W=20; N=300; lr="3e-5"
seed=101; W=20; N=300; lr="3e-5" 
seed=103; W=20; N=300; lr="3e-5"
  CUDA_VISIBLE_DEVICES="" python3 main_training.py --application rl --horizon_length 1000 --gym_env_name "HalfCheetah-v4" --rl_init_type normal --rl_init_scale 0.03 --gradient_estimator TruncatedPESK --num_particles $N --num_gradient_estimators 1 --trunc_length $W --K 1000 --sigma 0.004 --noloss_normalize --outer_optimizer_specs "SGD||CONSTANT$lr" --outer_iterations 1000 --init_seed 1 --remaining_seed $seed --run_test_during_training --eval_freq 4
