# BulletArm
- [License](https://github.com/ColinKohler/BulletArm/blob/main/LICENSE)

**BulletArm** is a benchmark and learning-environment for robotic manipulation. BulletArm provides a set of standardized 
benchmark tasks in simulation alongside a collection of baseline algorithms. Please see our paper [here]().

## Table of Contents
1. [Requirments](#requirments)
2. [Installation](#install)
4. [Parameters](#parameters)
1. [Benchmarks](#benchmarks)
5. [Publications](#publications)

<a name="requirments"></a>
## Requirments

<a name="install"></a>
## Install
1. Install Python 3.7
2. Clone this repo
    ```
    git clone https://github.com/ColinKohler/helping_hands_rl_envs.git
    cd helping_hands_rl_envs
    ```
3. Install dependencies
    ```
    pip install -r requirements.txt 
    ```
4. Install this package
    ```
    pip install .
    ```
5. Run the block stacking demo
    ```python
    from bulletarm import env_factory
    # environment parameters
    env_config = {'render': True}
    # create 1 block_stacking environment
    env = env_factory.createEnvs(1, 'block_stacking', env_config)
    env.reset()
    for i in range(5, -1, -1):
        action = env.getNextAction()
        (states_, in_hands_, obs_), rewards, dones = env.step(action)
        input('press enter to continue')
    env.close()
    ```
<a name="benchmarks"></a>
## Benchmarks
### Open-Loop Benchmarks
#### Prerequisite
1. Install [PyTorch](https://pytorch.org/) (Recommended: pytorch==1.7.0, torchvision==0.8.1)
1. (Optional, required for 6D benchmark) Install [CuPy](https://github.com/cupy/cupy)
1. Install other required packages
    ```
    pip install -r baseline_requirement.txt
    ```
1. Goto the baseline directory
    ```
    cd helping_hands_rl_envs/helping_hands_rl_baselines/fc_dqn/scripts
    ```
#### Open-Loop 3D Benchmark
```
python main.py --algorithm=[algorithm] --architecture=[architecture] --env=[env]
```
- Select `[algorithm]` from: `sdqfd` (recommended), `dqfd`, `adet`, `dqn`
- Select `[architecture]` from: `equi_asr` (recommended), `cnn_asr`, `equi_fcn`, `cnn_fcn`, `rot_fcn`
- Add `--fill_buffer_deconstruct` to use deconstruction planner for gathering expert data.
#### Open-Loop 2D Benchmark
```
python main.py  --algorithm=[algorithm] --architecture=[architecture] --action_sequence=xyp --random_orientation=f --env=[env]
```
- Select `[algorithm]` from: `sdqfd` (recommended), `dqfd`, `adet`, `dqn`
- Select `[architecture]` from: `equi_fcn` (recommended), `cnn_fcn`
- Add `--fill_buffer_deconstruct` to use deconstruction planner for gathering expert data.
#### Open-Loop 6D Benchmark
```
python main.py  --algorithm=[algorithm] --architecture=[architecture] --action_sequence=xyzrrrp --patch_size=[patch_size] --env=[env]
```
- Select `[algorithm]` from: `sdqfd` (recommended), `dqfd`, `adet`, `dqn`
- Select `[architecture]` from: `equi_deictic_asr` (recommended), `cnn_asr`
- Set `[patch_size]` to be `40` (required for `bumpy_box_palletizing` environment) or `24`
- Add `--fill_buffer_deconstruct` to use deconstruction planner for gathering expert data.

#### Additional Training Arguments
See [bulletarm_baselines/fc_dqn/utils/parameters.py](bulletarm_baselines/fc_dqn/utils/parameters.py)

### Close-Loop Benchmarks
#### Prerequisite
1. Install [PyTorch](https://pytorch.org/) (Recommended: pytorch==1.7.0, torchvision==0.8.1)
1. (Optional, required for 6D benchmark) Install [CuPy](https://github.com/cupy/cupy)
1. Install other required packages
    ```
    pip install -r baseline_requirement.txt
    ```
1. Goto the baseline directory
    ```
    cd helping_hands_rl_envs/helping_hands_rl_baselines/equi_rl/scripts
    ```
#### Close-Loop 4D Benchmark
```
python main.py --algorithm=[algorithm] --env=[env]
```
- Select `[algorithm]` from: `sac`, `sacfd`, `equi_sac`, `equi_sacfd`, `ferm_sac`, `ferm_sacfd`, `rad_sac`, `rad_sacfd`, `drq_sac`, `drq_sacfd`
- To use PER and data augmentation buffer, add `--buffer=per_expert_aug`
#### Close-Loop 3D Benchmark
```
python main.py --algorithm=[algorithm] --action_sequence=pxyz --random_orientation=f --env=[env]
```
- Select `[algorithm]` from: `sac`, `sacfd`, `equi_sac`, `equi_sacfd`, `ferm_sac`, `ferm_sacfd`, `rad_sac`, `rad_sacfd`, `drq_sac`, `drq_sacfd`

#### Additional Training Arguments
See [bulletarm_baselines/equi_rl/utils/parameters.py](bulletarm_baselines/equi_rl/utils/parameters.py)

<a name="publications"></a>
## Publications
If you like this package and use it in your own work, please cite this repository.
