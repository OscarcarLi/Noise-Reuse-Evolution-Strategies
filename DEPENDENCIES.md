[comment]: # (conda remove --name nres_env --all)

# Dependencies

We use `conda` to manage all the package dependencies used for this project through a conda environment.

### Step 1. creating a conda env named `nres_env`
```bash
conda create --name nres_env python=3.9
conda activate nres_env
```

### Step 2. install JAX with GPU support (this step might be different for different machines)
```
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
To check that the installed `jax` package can use GPUs, use the line in the terminal (after activating the conda env `nres_env`):
```bash
python3 -c "import jax; print(jax.devices())"
```

### Step 3. install JAX libraries
```bash
# jax neural network libraries
pip install dm-haiku
pip install flax
# jax optimization library
pip install optax
```


### Step 4. install tensorflow (we only need tensorflow for tensorboard logging)
```bash
conda install -c conda-forge tensorflow
```
We don't need to find GPUs for tensorflow as we only use it for tensorboard logging
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```
In addition, we'll need `tensorflow_datasets` to load data.
```bash
pip install tensorflow-datasets
```

### Step 5. install `gymnasium` (reinforcement learning library)
```bash
pip install "gymnasium[mujoco]"
# downgrading mujoco fix some stale xml files for some mujoco envs
pip install mujoco==2.3.3
```
To check `gymnasium` is properly installed,
```bash
python3 -c "import gymnasium as gym; env = gym.make('Swimmer-v4'); print(env)"
python3 -c "import gymnasium as gym; env = gym.make('HalfCheetah-v4'); print(env)"
```

### Step 6. Miscellaneous packages
```bash
pip install absl-py # for command line parsing
pip install pytz # for time stamping runs
pip install tqdm # for progress bars
pip install ipdb # for debugging
python -m pip install -U matplotlib
pip install jupyterlab
```