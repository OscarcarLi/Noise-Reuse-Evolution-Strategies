remaining_seed_list=(53 59 61 67 71)
for seed in "${remaining_seed_list[@]}"
do
# copy the code for each gradient estimator method below to run training sequentially
done

# BPTT training
seed=53 # prime numbers 53 59 61 67 71
device=0
CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'fashion_mnist' --image_length 8 --lopt_batch_size 128 --use_val --nouse_accuracy --lopt_inner_model mlp --lopt_activation gelu --lopt_hidden_dims "32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 88 --gradient_estimator FullGradient --num_particles 1 --num_gradient_estimators 1 --outer_optimizer_specs "ADAM||CONSTANT3e-4"  --outer_iterations 2000 --init_seed 2 --remaining_seed $seed --norun_test_during_training --run_test_after_training --eval_freq 20

# TBPTT training
seed=53 # prime numbers 53 59 61 67 71
device=1
CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'fashion_mnist' --image_length 8 --lopt_batch_size 128 --use_val --nouse_accuracy --lopt_inner_model mlp --lopt_activation gelu --lopt_hidden_dims "32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 88 --gradient_estimator TruncatedBPTT --num_particles 100 --num_gradient_estimators 1 --trunc_length 1 --outer_optimizer_specs "SGD||CONSTANT1e-2" --outer_iterations 320000 --init_seed 2 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 100 --eval_watershed 20000 --second_eval_freq 1000

# DODGE training
seed=53 # prime numbers 53 59 61 67 71
device=2
CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'fashion_mnist' --image_length 8 --lopt_batch_size 128 --use_val --nouse_accuracy --lopt_inner_model mlp --lopt_activation gelu --lopt_hidden_dims "32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 88 --gradient_estimator TruncatedForwardModeK --num_particles 100 --num_gradient_estimators 1 --trunc_length 1 --K 1001 --outer_optimizer_specs "ADAM||CONSTANT1e-5" --outer_iterations 290000 --init_seed 2 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 100 --eval_watershed 20000 --second_eval_freq 1000

# UORO training
seed=53 # prime numbers 53 59 61 67 71
device=3
CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'fashion_mnist' --image_length 8 --lopt_batch_size 128 --use_val --nouse_accuracy --lopt_inner_model mlp --lopt_activation gelu --lopt_hidden_dims "32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 88 --gradient_estimator UORO --num_particles 100 --num_gradient_estimators 1 --trunc_length 1 --outer_optimizer_specs "ADAM||CONSTANT3e-5" --outer_iterations 260000 --init_seed 2 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 100 --eval_watershed 20000 --second_eval_freq 1000

# FullES training
seed=53 # prime numbers 53 59 61 67 71
device=4
CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'fashion_mnist' --image_length 8 --lopt_batch_size 128 --use_val --nouse_accuracy --lopt_inner_model mlp --lopt_activation gelu --lopt_hidden_dims "32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 88  --gradient_estimator FullES --num_particles 10 --num_gradient_estimators 1 --sigma 0.01 --outer_optimizer_specs "ADAM||CONSTANT3e-2" --outer_iterations 5000 --init_seed 2 --remaining_seed $seed --norun_test_during_training --run_test_after_training --eval_freq 20

# TES training
seed=53 # prime numbers 53 59 61 67 71
device=5
CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'fashion_mnist' --image_length 8 --lopt_batch_size 128 --use_val --nouse_accuracy --lopt_inner_model mlp --lopt_activation gelu --lopt_hidden_dims "32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 88 --gradient_estimator TruncatedESBiased --num_particles 100 --num_gradient_estimators 1 --trunc_length 1 --sigma 0.01 --outer_optimizer_specs "ADAM||CONSTANT1e-3" --outer_iterations 180000 --init_seed 2 --remaining_seed $seed --norun_test_during_training --run_test_after_training --eval_freq 100 --eval_watershed 20000 --second_eval_freq 1000

# PES training
seed=53 # prime numbers 53 59 61 67 71
device=6
CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'fashion_mnist' --image_length 8 --lopt_batch_size 128 --use_val --nouse_accuracy --lopt_inner_model mlp --lopt_activation gelu --lopt_hidden_dims "32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 88 --gradient_estimator TruncatedPESK --num_particles 100 --num_gradient_estimators 1 --trunc_length 1 --K 1 --sigma 0.01 --outer_optimizer_specs "ADAM||CONSTANT3e-4" --outer_iterations 240000 --init_seed 2 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 100 --eval_watershed 20000 --second_eval_freq 1000

# NRES training
seed=53 # prime numbers 53 59 61 67 71
device=7
CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'fashion_mnist' --image_length 8 --lopt_batch_size 128 --use_val --nouse_accuracy --lopt_inner_model mlp --lopt_activation gelu --lopt_hidden_dims "32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 88 --gradient_estimator TruncatedPESK --num_particles 100 --num_gradient_estimators 1 --trunc_length 1 --K 1001 --sigma 0.01 --outer_optimizer_specs "ADAM||CONSTANT3e-4" --outer_iterations 240000 --init_seed 2 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 100 --eval_watershed 20000 --second_eval_freq 1000

# GPESK with different values of K
K=4 # 4 16 64 256
seed=53 # prime numbers 53 59 61 67 71
device=3
CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'fashion_mnist' --image_length 8 --lopt_batch_size 128 --use_val --nouse_accuracy --lopt_inner_model mlp --lopt_activation gelu --lopt_hidden_dims "32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 88 --gradient_estimator TruncatedPESK --num_particles 100 --num_gradient_estimators 1 --trunc_length 1 --K $K --sigma 0.01 --outer_optimizer_specs "ADAM||CONSTANT3e-4" --outer_iterations 240000 --init_seed 2 --remaining_seed $seed --norun_test_during_training -run_test_after_training --eval_freq 100 --eval_watershed 20000 --second_eval_freq 1000