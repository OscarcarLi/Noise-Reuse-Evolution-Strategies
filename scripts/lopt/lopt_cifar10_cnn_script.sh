# prime numbers 23, 29, 31, 37, 41
remaining_seed_list=(23 29 31 37 41)


# NRES run test during training
seed=23
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="3.0e-4"; device=1; outer_iterations=20000
seed=29
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="3.0e-4"; device=3; outer_iterations=20000
seed=31
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="3.0e-4"; device=7; outer_iterations=20000
seed=37
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="3.0e-4"; device=6; outer_iterations=20000
seed=41
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="3.0e-4"; device=7; outer_iterations=20000

# set a seed and its config in command line before running the following command
  CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'cifar10' --image_length 32 --lopt_batch_size $batch_size --use_val --nouse_accuracy --use_threshold_outer_cross_entropy_loss --lopt_inner_model cnn --lopt_activation $activation --lopt_hidden_channels "32,32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 66 --gradient_estimator TruncatedPESK --num_particles $N --num_gradient_estimators 1 --trunc_length $W --K 1001 --sigma 0.01 --outer_optimizer_specs "ADAM||CONSTANT$lr" --outer_iterations $outer_iterations --init_seed $init_seed --remaining_seed $seed --run_test_during_training --norun_test_after_training --eval_freq 10;

# PES
seed=23
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="3.0e-4"; device=3; outer_iterations=20000
seed=29
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="3.0e-4"; device=4; outer_iterations=20000
seed=31
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="3.0e-4"; device=2; outer_iterations=20000
seed=37
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="3.0e-4"; device=0; outer_iterations=20000
seed=41
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="3.0e-4"; device=1; outer_iterations=20000

  CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'cifar10' --image_length 32 --lopt_batch_size $batch_size --use_val --nouse_accuracy --use_threshold_outer_cross_entropy_loss --lopt_inner_model cnn --lopt_activation $activation --lopt_hidden_channels "32,32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 66 --gradient_estimator TruncatedPESK --num_particles $N --num_gradient_estimators 1 --trunc_length $W --K $W --sigma 0.01 --outer_optimizer_specs "ADAM||CONSTANT$lr" --outer_iterations $outer_iterations --init_seed $init_seed --remaining_seed $seed --run_test_during_training --norun_test_after_training --eval_freq 10;

# TES
seed=23
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="1.0e-3"; device=5; outer_iterations=20000
seed=29
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="1.0e-3"; device=6; outer_iterations=20000
seed=31
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="1.0e-3"; device=3; outer_iterations=20000
seed=37
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="1.0e-3"; device=4; outer_iterations=20000
seed=41
batch_size=32; init_seed=5; N=10; W=100; activation="relu"; lr="1.0e-3"; device=5; outer_iterations=20000

  CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'cifar10' --image_length 32 --lopt_batch_size $batch_size --use_val --nouse_accuracy --use_threshold_outer_cross_entropy_loss --lopt_inner_model cnn --lopt_activation $activation --lopt_hidden_channels "32,32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 66 --gradient_estimator TruncatedESBiased --num_particles $N --num_gradient_estimators 1 --trunc_length $W --sigma 0.01 --noloss_normalize --outer_optimizer_specs "ADAM||CONSTANT$lr" --outer_iterations $outer_iterations --init_seed $init_seed --remaining_seed $seed --run_test_during_training --norun_test_after_training --eval_freq 10;

# FullES
seed=23
batch_size=32; init_seed=5; N=1; activation="relu"; lr="1.0e-3"; device=0; outer_iterations=20000
seed=29
batch_size=32; init_seed=5; N=1; activation="relu"; lr="1.0e-3"; device=1; outer_iterations=20000
seed=31
batch_size=32; init_seed=5; N=1; activation="relu"; lr="1.0e-3"; device=2; outer_iterations=20000
seed=37
batch_size=32; init_seed=5; N=1; activation="relu"; lr="1.0e-3"; device=6; outer_iterations=20000
seed=41
batch_size=32; init_seed=5; N=1; activation="relu"; lr="1.0e-3"; device=7; outer_iterations=20000

  CUDA_VISIBLE_DEVICES=$device python3 main_training.py --application lopt --horizon_length 1000 --lopt_dataset_name 'cifar10' --image_length 32 --lopt_batch_size $batch_size --use_val --nouse_accuracy --use_threshold_outer_cross_entropy_loss --lopt_inner_model cnn --lopt_activation $activation --lopt_hidden_channels "32,32,32,32" --lopt_architecture MLPLOpt --lopt_single_sequence_seed 66 --gradient_estimator FullES --num_particles $N --num_gradient_estimators 1 --sigma 0.01 --outer_optimizer_specs "ADAM||CONSTANT$lr" --outer_iterations $outer_iterations --init_seed $init_seed --remaining_seed $seed --run_test_during_training --norun_test_after_training --eval_freq 10;
