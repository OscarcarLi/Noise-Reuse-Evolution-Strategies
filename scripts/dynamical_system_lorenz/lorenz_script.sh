# BPTT training
seed=101 # prime numbers 101, 103, 107, 109, 113
CUDA_VISIBLE_DEVICES=0 python3 main_training.py --application lorenz --horizon_length 2000 --lorenz_dt 0.005 --lorenz_init_loga 3.116 --lorenz_init_logr 3.7 --lorenz_single_init --gradient_estimator FullGradient --num_particles 1 --num_gradient_estimators 1 --outer_optimizer_specs "SGD||CONSTANT1e-8" --outer_iterations 1000 --init_seed 1 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 10

# TBPTT training
seed=101 # prime numbers 101, 103, 107, 109, 113
CUDA_VISIBLE_DEVICES=1 python3 main_training.py --application lorenz --horizon_length 2000 --lorenz_dt 0.005 --lorenz_init_loga 3.116 --lorenz_init_logr 3.7 --lorenz_single_init --gradient_estimator TruncatedBPTT --num_particles 200 --num_gradient_estimators 1 --trunc_length 100 --outer_optimizer_specs "SGD||CONSTANT3e-4" --outer_iterations 10000 --init_seed 1 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 10

# UORO training
seed=101 # prime numbers 101, 103, 107, 109, 113
CUDA_VISIBLE_DEVICES=2 python3 main_training.py --application lorenz --horizon_length 2000 --lorenz_dt 0.005 --lorenz_init_loga 3.116 --lorenz_init_logr 3.7 --lorenz_single_init --gradient_estimator UORO --num_particles 200 --num_gradient_estimators 1 --trunc_length 100 --outer_optimizer_specs "SGD||CONSTANT1e-13" --outer_iterations 5000 --init_seed 1 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 10

# DODGE training
seed=101 # prime numbers 101, 103, 107, 109, 113
CUDA_VISIBLE_DEVICES=3 python3 main_training.py --application lorenz --horizon_length 2000 --lorenz_dt 0.005 --lorenz_init_loga 3.116 --lorenz_init_logr 3.7 --lorenz_single_init --gradient_estimator TruncatedForwardModeK --num_particles 200 --num_gradient_estimators 1 --trunc_length 100 --K 2000 --outer_optimizer_specs "SGD||CONSTANT1e-10" --outer_iterations 5000 --init_seed 1 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 10

# FullES training
seed=101 # prime numbers 101, 103, 107, 109, 113
CUDA_VISIBLE_DEVICES=4 python3 main_training.py --application lorenz --horizon_length 2000 --lorenz_dt 0.005 --lorenz_init_loga 3.116 --lorenz_init_logr 3.7 --lorenz_single_init --gradient_estimator FullES --num_particles 10 --num_gradient_estimators 1 --sigma 0.04 --outer_optimizer_specs "SGD||CONSTANT3e-5" --outer_iterations 2000 --init_seed 1 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 10

# TES training
seed=101 # prime numbers 101, 103, 107, 109, 113
CUDA_VISIBLE_DEVICES=5 python3 main_training.py --application lorenz --horizon_length 2000 --lorenz_dt 0.005 --lorenz_init_loga 3.116 --lorenz_init_logr 3.7 --lorenz_single_init --gradient_estimator TruncatedESBiased --num_particles 200 --num_gradient_estimators 1 --trunc_length 100 --sigma 0.04 --outer_optimizer_specs "SGD||CONSTANT3e-4" --outer_iterations 5000 --init_seed 1 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 10

# PES training
seed=101 # prime numbers 101, 103, 107, 109, 113
CUDA_VISIBLE_DEVICES=6 python3 main_training.py --application lorenz --horizon_length 2000 --lorenz_dt 0.005 --lorenz_init_loga 3.116 --lorenz_init_logr 3.7 --lorenz_single_init --gradient_estimator TruncatedPESK --num_particles 200 --num_gradient_estimators 1 --trunc_length 100 --K 100 --sigma 0.04 --outer_optimizer_specs "SGD||PIECEWISE_CONSTANT1e-5(1000,0.1)" --outer_iterations 10000 --init_seed 1 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 10

# NRES training
seed=101 # prime numbers 101, 103, 107, 109, 113
CUDA_VISIBLE_DEVICES=7 python3 main_training.py --application lorenz --horizon_length 2000 --lorenz_dt 0.005 --lorenz_init_loga 3.116 --lorenz_init_logr 3.7 --lorenz_single_init --gradient_estimator TruncatedPESK --num_particles 200 --num_gradient_estimators 1 --trunc_length 100 --K 2001 --sigma 0.04 --outer_optimizer_specs "SGD||CONSTANT1e-5" --outer_iterations 5000 --init_seed 1 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 10