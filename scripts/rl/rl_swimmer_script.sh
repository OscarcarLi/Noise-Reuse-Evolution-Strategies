# training NRES for 5 different random seeds.
remaining_seed_list=(23 29 31 37 41)

# FullES
for seed in "${remaining_seed_list[@]}"
do
  CUDA_VISIBLE_DEVICES="" python3 main_training.py --application rl --horizon_length 1000 --gym_env_name "Swimmer-v4" --rl_init_type zero --gradient_estimator FullES --num_particles 3 --num_gradient_estimators 1 --sigma 0.3 --noloss_normalize --outer_optimizer_specs "SGD||CONSTANT1e0" --outer_iterations 200 --init_seed 0 --remaining_seed $seed --run_test_during_training --eval_freq 1
done

# TES
for seed in "${remaining_seed_list[@]}"
do
  CUDA_VISIBLE_DEVICES="" python3 main_training.py --application rl --horizon_length 1000 --gym_env_name "Swimmer-v4" --rl_init_type zero --gradient_estimator TruncatedESBiased --num_particles 30 --num_gradient_estimators 1 --trunc_length 100 --sigma 0.3 --noloss_normalize --outer_optimizer_specs "SGD||CONSTANT3e1" --outer_iterations 200 --init_seed 0 --remaining_seed $seed --run_test_during_training --eval_freq 1
done

# PES
for seed in "${remaining_seed_list[@]}"
do
  CUDA_VISIBLE_DEVICES="" python3 main_training.py --application rl --horizon_length 1000 --gym_env_name "Swimmer-v4" --rl_init_type zero --gradient_estimator TruncatedPESK --num_particles 30 --num_gradient_estimators 1 --trunc_length 100 --K 100 --sigma 0.3 --noloss_normalize --outer_optimizer_specs "SGD||CONSTANT1e0" --outer_iterations 200 --init_seed 0 --remaining_seed $seed --run_test_during_training --eval_freq 1
done

# NRES
for seed in "${remaining_seed_list[@]}"
do
  CUDA_VISIBLE_DEVICES="" python3 main_training.py --application rl --horizon_length 1000 --gym_env_name "Swimmer-v4" --rl_init_type zero --gradient_estimator TruncatedPESK --num_particles 30 --num_gradient_estimators 1 --trunc_length 100 --K 1000 --sigma 0.3 --noloss_normalize --outer_optimizer_specs "SGD||CONSTANT3e0" --outer_iterations 200 --init_seed 0 --remaining_seed $seed --run_test_during_training --eval_freq 1;
done