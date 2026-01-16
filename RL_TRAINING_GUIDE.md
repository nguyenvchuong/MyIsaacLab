# RL Training Pipeline Guide for Isaac Lab

This guide explains the key functions and flow when training RL agents in Isaac Lab, specifically for `Isaac-Velocity-Rough-Digit-v0`.

## 📋 Table of Contents
1. [Training Entry Point](#1-training-entry-point)
2. [Environment Creation](#2-environment-creation)
3. [Managers System](#3-managers-system)
4. [RL Training Loop](#4-rl-training-loop)
5. [Key Files to Study](#5-key-files-to-study)

---

## 1. Training Entry Point

### Main Script: `scripts/reinforcement_learning/rsl_rl/train.py`

**Key Function: `main()`** (line 115)
- Entry point for training
- Sets up configuration, environment, and runner
- Calls `runner.learn()` to start training

**Flow:**
```python
1. Parse CLI arguments → hydra_task_config decorator loads configs
2. Create environment → gym.make(args_cli.task, cfg=env_cfg)
3. Wrap environment → RslRlVecEnvWrapper(env)
4. Create runner → OnPolicyRunner(env, agent_cfg, log_dir, device)
5. Start training → runner.learn(num_learning_iterations)
```

**Key Configuration Files:**
- **Environment Config**: `source/isaaclab_tasks/.../digit/rough_env_cfg.py`
- **Agent Config**: `source/isaaclab_tasks/.../digit/agents/rsl_rl_ppo_cfg.py`

---

## 2. Environment Creation

### Class: `ManagerBasedRLEnv` 
**Location**: `source/isaaclab/isaaclab/envs/manager_based_rl_env.py`

**Key Methods:**

#### `__init__(cfg)` - Environment Initialization
- Creates the simulation scene (robots, terrain, sensors)
- Initializes managers: Observation, Action, Reward, Termination, Command, Event
- Sets up buffers for observations, rewards, resets

#### `reset(seed, env_ids, options)` - Reset Environments
- Resets specified environments to initial state
- Applies randomization events (domain randomization)
- Computes initial observations
- **Called once at the start**, then handled automatically by `step()`

#### `step(action)` - Environment Step (CRITICAL!)
**Location**: `source/isaaclab/isaaclab/envs/manager_based_rl_env.py:154`

**Flow:**
```python
def step(self, action: torch.Tensor):
    # 1. Process actions through ActionManager
    self.action_manager.process_action(action)
    
    # 2. Apply actions to simulation
    self.scene.write_data_to_sim()
    
    # 3. Step physics simulation (decimation times)
    for _ in range(self.decimation):
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
    
    # 4. Compute rewards
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    
    # 5. Check terminations
    self.reset_terminated, self.reset_time_outs = self.termination_manager.compute()
    
    # 6. Reset terminated environments
    if len(reset_env_ids) > 0:
        self._reset_idx(reset_env_ids)
    
    # 7. Update commands (for velocity tracking tasks)
    self.command_manager.compute(dt=self.step_dt)
    
    # 8. Compute observations
    self.obs_buf = self.observation_manager.compute(update_history=True)
    
    # 9. Return (obs, reward, terminated, timeout, extras)
    return self.obs_buf, self.reward_buf, self.reset_terminated, ...
```

---

## 3. Managers System

The manager-based workflow uses specialized managers for different aspects:

### 3.1 Observation Manager
**Location**: `source/isaaclab/isaaclab/managers/observation_manager.py`

**Purpose**: Constructs observation vectors from simulation state

**For Digit task** (`rough_env_cfg.py:136-178`):
```python
Observations include:
- base_lin_vel: Linear velocity of robot base
- base_ang_vel: Angular velocity of robot base  
- projected_gravity: Gravity vector in robot frame
- velocity_commands: Desired velocity commands
- joint_pos: Joint positions (relative to default)
- joint_vel: Joint velocities
- actions: Previous action (for action history)
- height_scan: Height scanner readings (terrain awareness)
```

**Key Function**: `observation_manager.compute(update_history=True)`
- Calls each observation term function
- Applies noise (if enabled)
- Concatenates into observation vector
- Shape: `(num_envs, obs_dim)`

### 3.2 Action Manager
**Location**: `source/isaaclab/isaaclab/managers/action_manager.py`

**Purpose**: Processes raw actions from policy into joint commands

**For Digit task** (`rough_env_cfg.py:200-208`):
```python
Actions: Joint position targets for all leg and arm joints
- Scale: 0.5 (actions are scaled before applying)
- Uses default offset (neutral pose)
```

**Key Function**: `action_manager.process_action(action)`
- Scales actions
- Converts to joint position targets
- Applies to robot joints

### 3.3 Reward Manager ⭐ MOST IMPORTANT
**Location**: `source/isaaclab/isaaclab/managers/reward_manager.py`

**Purpose**: Computes reward signal from multiple reward terms

**Key Function**: `reward_manager.compute(dt)` (line 128)
```python
def compute(self, dt: float) -> torch.Tensor:
    self._reward_buf[:] = 0.0
    for name, term_cfg in zip(self._term_names, self._term_cfgs):
        # Compute each reward term
        value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
        self._reward_buf += value
    return self._reward_buf  # Shape: (num_envs,)
```

**For Digit task** (`rough_env_cfg.py:19-132`):
```python
Reward Terms:
1. termination_penalty: -100.0 (penalty for falling/terminating
2. track_lin_vel_xy_exp: 1.0 (reward for tracking desired linear velocity)
3. track_ang_vel_z_exp: 1.0 (reward for tracking desired angular velocity)
4. feet_air_time: 0.25 (reward for keeping feet in air during swing)
5. feet_slide: -0.25 (penalty for feet sliding on ground)
6. dof_torques_l2: -1.0e-6 (penalty for high joint torques)
7. dof_acc_l2: -2.0e-7 (penalty for high joint accelerations)
8. action_rate_l2: -0.008 (penalty for rapid action changes)
9. flat_orientation_l2: -2.5 (penalty for non-upright orientation)
10. stand_still: -0.4 (penalty when standing but command is zero)
11. lin_vel_z_l2: -2.0 (penalty for vertical velocity)
12. ang_vel_xy_l2: -0.1 (penalty for unwanted angular velocities)
... and more
```

**Reward Functions Location**: 
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp.py`
- Functions like `track_lin_vel_xy_yaw_frame_exp`, `feet_air_time_positive_biped`, etc.

### 3.4 Termination Manager
**Location**: `source/isaaclab/isaaclab/managers/termination_manager.py`

**Purpose**: Determines when episodes should terminate

**For Digit task** (`rough_env_cfg.py:182-196`):
```python
Termination Conditions:
1. time_out: Episode reaches max length
2. base_contact: Torso touches ground (illegal contact)
3. base_orientation: Robot tilts too much (>0.7 radians)
```

### 3.5 Command Manager
**Location**: `source/isaaclab/isaaclab/managers/command_manager.py`

**Purpose**: Generates velocity commands for the robot to track

**For Digit task**: Generates random velocity commands (lin_vel_x, lin_vel_y, ang_vel_z)
- Resampled every 3-8 seconds
- Robot must learn to track these commands

### 3.6 Event Manager
**Location**: `source/isaaclab/isaaclab/managers/event_manager.py`

**Purpose**: Domain randomization and episodic events

**For Digit task**:
- Random base mass
- Random external forces/torques
- Random initial joint positions
- Push events

---

## 4. RL Training Loop

### RSL-RL Runner: `OnPolicyRunner`
**Location**: External library `rsl_rl.runners.OnPolicyRunner`

**Key Method**: `runner.learn(num_learning_iterations)`

**Training Loop (simplified)**:
```python
for iteration in range(num_learning_iterations):
    # 1. Collect rollouts (interact with environment)
    for step in range(num_steps_per_env):
        # Get action from policy
        actions = policy(observations)
        
        # Step environment
        obs, rewards, dones, _ = env.step(actions)
        
        # Store in replay buffer
        buffer.add(obs, actions, rewards, dones, ...)
    
    # 2. Compute advantages (GAE - Generalized Advantage Estimation)
    advantages = compute_gae(rewards, values, dones, gamma, lam)
    
    # 3. Update policy (PPO update)
    for epoch in range(num_learning_epochs):
        for batch in batches:
            # Compute policy loss (clipped PPO objective)
            policy_loss = compute_policy_loss(batch, advantages)
            
            # Compute value loss
            value_loss = compute_value_loss(batch)
            
            # Compute entropy bonus
            entropy = compute_entropy(policy(batch.obs))
            
            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            # Backprop and update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
    
    # 4. Log metrics and save checkpoint
    if iteration % save_interval == 0:
        save_checkpoint(f"model_{iteration}.pt")
```

**Key Hyperparameters** (from `rsl_rl_ppo_cfg.py`):
- `num_steps_per_env`: 24 (steps per environment per update)
- `num_learning_epochs`: 5 (PPO update epochs)
- `num_mini_batches`: 4 (batch size for updates)
- `learning_rate`: 1.0e-3
- `gamma`: 0.99 (discount factor)
- `lam`: 0.95 (GAE lambda)
- `clip_param`: 0.2 (PPO clipping)
- `entropy_coef`: 0.01 (entropy bonus)

---

## 5. Key Files to Study

### Environment Configuration
1. **`source/isaaclab_tasks/.../digit/rough_env_cfg.py`**
   - Defines reward terms, observations, terminations, actions
   - **Study this to understand the task design**

2. **`source/isaaclab_tasks/.../locomotion/velocity/mdp.py`**
   - Contains all reward/observation/termination functions
   - **Study this to understand reward shaping**

### Environment Implementation
3. **`source/isaaclab/isaaclab/envs/manager_based_rl_env.py`**
   - Core environment class
   - **Study `step()` method (line 154) - this is the heart of the environment**

4. **`source/isaaclab/isaaclab/envs/manager_based_env.py`**
   - Base environment with managers
   - **Study managers initialization and usage**

### Managers
5. **`source/isaaclab/isaaclab/managers/reward_manager.py`**
   - How rewards are computed
   - **Study `compute()` method (line 128)**

6. **`source/isaaclab/isaaclab/managers/observation_manager.py`**
   - How observations are constructed

7. **`source/isaaclab/isaaclab/managers/action_manager.py`**
   - How actions are processed

### Training
8. **`scripts/reinforcement_learning/rsl_rl/train.py`**
   - Training script entry point
   - **Study the main() function flow**

9. **`source/isaaclab_tasks/.../digit/agents/rsl_rl_ppo_cfg.py`**
   - PPO hyperparameters
   - **Study to understand RL algorithm settings**

### Agent Configuration
10. **`source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`**
    - Base configuration classes
    - Policy and algorithm configs

---

## 🔍 How to Trace Execution

When you run `python train.py --task=Isaac-Velocity-Rough-Digit-v0 --num_envs=128 --headless`:

1. **Start**: `train.py:main()` (line 115)
2. **Environment Creation**: `gym.make()` → Creates `ManagerBasedRLEnv`
3. **Initialization**: `env.__init__()` → Sets up managers and scene
4. **First Reset**: `env.reset()` → Initializes all environments
5. **Training Loop**: `runner.learn()` → Calls `env.step()` repeatedly
6. **Each Step**: 
   - Policy outputs action
   - `env.step(action)` → Processes action, steps sim, computes reward/obs
   - Data stored in buffer
7. **Update**: After collecting rollouts, PPO updates policy
8. **Repeat**: Until `max_iterations` reached

---

## 💡 Key Concepts

1. **Vectorized Environments**: All operations happen in parallel for `num_envs` environments
2. **Decimation**: Multiple physics steps per environment step (for stability)
3. **Reward Shaping**: Multiple reward terms combined to guide learning
4. **Domain Randomization**: Events manager applies randomization for robustness
5. **Command Tracking**: Robot learns to track randomly generated velocity commands
6. **PPO Algorithm**: On-policy algorithm that updates from collected rollouts

---

## 🎯 What to Modify

- **Reward Function**: Edit `rough_env_cfg.py` → `DigitRewards` class
- **Observations**: Edit `rough_env_cfg.py` → `DigitObservations` class  
- **Termination Conditions**: Edit `rough_env_cfg.py` → `TerminationsCfg` class
- **Actions**: Edit `rough_env_cfg.py` → `ActionsCfg` class
- **RL Hyperparameters**: Edit `agents/rsl_rl_ppo_cfg.py` → `DigitRoughPPORunnerCfg` class
- **Environment Settings**: Edit `rough_env_cfg.py` → `DigitRoughEnvCfg` class

---

## 📚 Additional Resources

- RSL-RL Documentation: Check the rsl-rl library for PPO implementation details
- Isaac Lab Docs: `docs/source/tutorials/03_envs/` for environment creation tutorials
- MDP Functions: `source/isaaclab_tasks/.../locomotion/velocity/mdp.py` for reward/observation functions










