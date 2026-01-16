# Learning Cartpole Environment - Complete Guide

This guide explains the `cartpole_env.py` file, which is a simple example of the **Direct Workflow** in Isaac Lab.

---

## 🎯 What is Cartpole?

**Cartpole** is a classic control problem:
- A **cart** can move left/right on a track
- A **pole** is attached to the cart and can fall
- **Goal:** Keep the pole balanced upright by moving the cart
- **Action:** Apply force to the cart (left or right)
- **Observation:** Pole angle, pole velocity, cart position, cart velocity

---

## 📋 File Structure Overview

```python
cartpole_env.py
├── CartpoleEnvCfg (Configuration class)
│   ├── Environment settings (episode length, decimation)
│   ├── Simulation settings (dt, render_interval)
│   ├── Robot settings (cartpole configuration)
│   ├── Scene settings (num_envs, spacing)
│   └── Reward scales
│
└── CartpoleEnv (Environment class)
    ├── __init__() - Setup
    ├── _setup_scene() - Create physics scene
    ├── _pre_physics_step() - Process actions
    ├── _apply_action() - Apply forces to cart
    ├── _get_observations() - Compute observations
    ├── _get_rewards() - Compute rewards
    ├── _get_dones() - Check termination
    └── _reset_idx() - Reset environments
```

---

## 🔍 Part 1: Configuration Class (Lines 24-57)

### What is a Configuration Class?

A **configuration class** stores all the **settings** for your environment. Think of it as a blueprint.

```python
@configclass
class CartpoleEnvCfg(DirectRLEnvCfg):
    # This class defines all the settings for the Cartpole environment
```

### Key Settings Explained

#### Environment Settings (Lines 26-32)
```python
decimation = 2              # Skip 2 physics steps per action (faster simulation)
episode_length_s = 5.0      # Episode lasts 5 seconds
action_scale = 100.0        # Multiply action by 100 to get force in Newtons
action_space = 1            # 1D action (force on cart)
observation_space = 4       # 4D observation (pole angle, pole vel, cart pos, cart vel)
state_space = 0             # No state space (not used here)
```

**What they mean:**
- **decimation**: Instead of applying action every physics step, apply it every 2 steps (2x faster)
- **episode_length_s**: Each training episode lasts 5 seconds
- **action_scale**: Actions are typically [-1, 1], scaled to [-100, 100] Newtons

#### Simulation Settings (Line 35)
```python
sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
```

- **dt = 1/120**: Physics timestep = 0.0083 seconds (120 Hz)
- **render_interval**: Update visualization every 2 steps

#### Robot Settings (Line 38)
```python
robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
```

- Loads the cartpole robot model
- `prim_path`: Where to place robots in the scene (one per environment)

#### Scene Settings (Lines 43-45)
```python
scene: InteractiveSceneCfg = InteractiveSceneCfg(
    num_envs=4096,              # Create 4096 parallel environments!
    env_spacing=4.0,             # Space between environments (meters)
    replicate_physics=True,      # Share physics between similar environments
    clone_in_fabric=True,        # Use GPU-accelerated cloning
)
```

**Why 4096 environments?**
- Train on many environments simultaneously
- Much faster than training on one environment
- More diverse experience for the agent

#### Reward Scales (Lines 51-56)
```python
rew_scale_alive = 1.0           # Reward for staying alive
rew_scale_terminated = -2.0     # Penalty for falling
rew_scale_pole_pos = -1.0       # Penalty for pole angle (want it upright = 0)
rew_scale_cart_vel = -0.01      # Small penalty for cart movement
rew_scale_pole_vel = -0.005     # Small penalty for pole movement
```

**Reward design:**
- Positive reward for staying alive
- Negative penalties for:
  - Falling (terminated)
  - Pole not upright
  - Moving too much (encourages stability)

---

## 🤖 Part 2: Environment Class (Lines 59-153)

### Class Definition
```python
class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg
```

- Inherits from `DirectRLEnv` (base class for direct workflow)
- `cfg`: Stores the configuration

### __init__() - Initialization (Lines 62-67)

```python
def __init__(self, cfg: CartpoleEnvCfg, render_mode: str | None = None, **kwargs):
    super().__init__(cfg, render_mode, **kwargs)  # Initialize base class
    
    # Find joint indices (which joints are cart and pole)
    self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
    self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
    
    # Store action scale
    self.action_scale = self.cfg.action_scale
    
    # Get references to joint data (for faster access)
    self.joint_pos = self.cartpole.data.joint_pos
    self.joint_vel = self.cartpole.data.joint_vel
```

**What happens:**
1. Initialize parent class (sets up simulation, etc.)
2. Find which joints are the cart and pole
3. Store action scale for later use
4. Get references to joint data (position, velocity)

### _setup_scene() - Create Physics Scene (Lines 72-85)

```python
def _setup_scene(self):
    # 1. Create the cartpole robot
    self.cartpole = Articulation(self.cfg.robot_cfg)
    
    # 2. Add ground plane (the track the cart moves on)
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    
    # 3. Clone environments (create 4096 copies)
    self.scene.clone_environments(copy_from_source=False)
    
    # 4. Filter collisions for CPU (if needed)
    if self.device == "cpu":
        self.scene.filter_collisions(global_prim_paths=[])
    
    # 5. Add robot to scene
    self.scene.articulations["cartpole"] = self.cartpole
    
    # 6. Add lighting
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
```

**What happens:**
1. Creates the cartpole robot
2. Adds ground (track)
3. Clones to create 4096 parallel environments
4. Adds robot to scene
5. Adds lighting for visualization

### _pre_physics_step() - Process Actions (Lines 87-88)

```python
def _pre_physics_step(self, actions: torch.Tensor) -> None:
    self.actions = self.action_scale * actions.clone()
```

**What happens:**
- Receives actions from RL agent (typically [-1, 1])
- Scales them by `action_scale` (100.0) to get force in Newtons
- Stores for use in `_apply_action()`

**Example:**
- Agent outputs: `action = 0.5`
- After scaling: `self.actions = 0.5 * 100.0 = 50.0` Newtons

### _apply_action() - Apply Forces (Lines 90-91)

```python
def _apply_action(self) -> None:
    self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
```

**What happens:**
- Applies the force to the cart joint
- This moves the cart left or right
- Physics simulation then updates everything

### _get_observations() - Compute Observations (Lines 93-104)

```python
def _get_observations(self) -> dict:
    obs = torch.cat(
        (
            self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),   # Pole angle
            self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),  # Pole velocity
            self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),  # Cart position
            self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),   # Cart velocity
        ),
        dim=-1,
    )
    observations = {"policy": obs}
    return observations
```

**What happens:**
1. Gets pole angle (position of pole joint)
2. Gets pole velocity (how fast pole is moving)
3. Gets cart position (where cart is on track)
4. Gets cart velocity (how fast cart is moving)
5. Concatenates into 4D vector
6. Returns as dictionary with key "policy"

**Observation shape:**
- For 4096 environments: `(4096, 4)`
- Each row: `[pole_angle, pole_vel, cart_pos, cart_vel]`

### _get_rewards() - Compute Rewards (Lines 106-119)

```python
def _get_rewards(self) -> torch.Tensor:
    total_reward = compute_rewards(
        self.cfg.rew_scale_alive,        # 1.0
        self.cfg.rew_scale_terminated,   # -2.0
        self.cfg.rew_scale_pole_pos,     # -1.0
        self.cfg.rew_scale_cart_vel,     # -0.01
        self.cfg.rew_scale_pole_vel,     # -0.005
        self.joint_pos[:, self._pole_dof_idx[0]],   # Pole angle
        self.joint_vel[:, self._pole_dof_idx[0]],   # Pole velocity
        self.joint_pos[:, self._cart_dof_idx[0]],   # Cart position
        self.joint_vel[:, self._cart_dof_idx[0]],   # Cart velocity
        self.reset_terminated,                       # Did it fall?
    )
    return total_reward
```

**Calls the reward function** (explained below).

### _get_dones() - Check Termination (Lines 121-128)

```python
def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    # Update joint data
    self.joint_pos = self.cartpole.data.joint_pos
    self.joint_vel = self.cartpole.data.joint_vel
    
    # Check timeout (episode reached max length)
    time_out = self.episode_length_buf >= self.max_episode_length - 1
    
    # Check if out of bounds
    out_of_bounds = torch.any(
        torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1
    )
    # Also check if pole fell too far
    out_of_bounds = out_of_bounds | torch.any(
        torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1
    )
    
    return out_of_bounds, time_out
```

**What happens:**
- **time_out**: Episode reached 5 seconds (success!)
- **out_of_bounds**: Either:
  - Cart moved too far (> 3.0 meters)
  - Pole fell too far (> 90 degrees)
- Returns two boolean tensors: `(terminated, time_out)`

### _reset_idx() - Reset Environments (Lines 130-152)

```python
def _reset_idx(self, env_ids: Sequence[int] | None):
    if env_ids is None:
        env_ids = self.cartpole._ALL_INDICES  # Reset all
    
    super()._reset_idx(env_ids)  # Reset base class
    
    # Get default joint positions
    joint_pos = self.cartpole.data.default_joint_pos[env_ids]
    
    # Add random noise to pole angle (for exploration)
    joint_pos[:, self._pole_dof_idx] += sample_uniform(
        self.cfg.initial_pole_angle_range[0] * math.pi,  # -0.25 * π
        self.cfg.initial_pole_angle_range[1] * math.pi,  # +0.25 * π
        joint_pos[:, self._pole_dof_idx].shape,
        joint_pos.device,
    )
    
    # Get default velocities (zero)
    joint_vel = self.cartpole.data.default_joint_vel[env_ids]
    
    # Get default root state (position, orientation, velocity)
    default_root_state = self.cartpole.data.default_root_state[env_ids]
    default_root_state[:, :3] += self.scene.env_origins[env_ids]  # Offset by environment position
    
    # Update data
    self.joint_pos[env_ids] = joint_pos
    self.joint_vel[env_ids] = joint_vel
    
    # Write to simulation
    self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
```

**What happens:**
1. Determines which environments to reset
2. Gets default positions/velocities
3. Adds random noise to pole angle (so each episode starts differently)
4. Updates internal data
5. Writes new state to simulation

**Why random noise?**
- Helps agent learn to balance from different starting positions
- More robust policy

---

## 🎁 Part 3: Reward Function (Lines 155-174)

```python
@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # Reward for staying alive
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    # ↑ 1.0 if alive, 0.0 if terminated
    
    # Penalty for falling
    rew_termination = rew_scale_terminated * reset_terminated.float()
    # ↑ -2.0 if terminated, 0.0 if alive
    
    # Penalty for pole angle (want it at 0 = upright)
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    # ↑ More negative as pole angle increases
    
    # Small penalty for cart movement
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    
    # Small penalty for pole movement
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    
    # Total reward
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward
```

**Reward breakdown:**
- **rew_alive**: +1.0 per step if alive
- **rew_termination**: -2.0 if fell
- **rew_pole_pos**: Penalty for pole not upright (quadratic)
- **rew_cart_vel**: Small penalty for moving cart
- **rew_pole_vel**: Small penalty for pole movement

**Total reward per step:**
- If alive and balanced: ~1.0
- If fell: -2.0
- If alive but pole tilted: 1.0 - (pole_angle²) - small_penalties

**Why this design?**
- Encourages staying alive (positive reward)
- Discourages falling (large penalty)
- Encourages stability (penalties for movement)

---

## 🔄 The Training Loop (How It All Works Together)

```
1. Reset
   └─> _reset_idx() - Initialize environments
   
2. For each step:
   a. Get observation
      └─> _get_observations() - Returns [pole_angle, pole_vel, cart_pos, cart_vel]
   
   b. Agent decides action
      └─> Policy network outputs force (e.g., 0.5)
   
   c. Apply action
      └─> _pre_physics_step() - Scale action (0.5 → 50.0 N)
      └─> _apply_action() - Apply force to cart
      └─> Physics simulation updates
   
   d. Compute reward
      └─> _get_rewards() - Calculate reward based on state
   
   e. Check if done
      └─> _get_dones() - Check if episode ended
   
   f. If done, reset
      └─> _reset_idx() - Start new episode
```

---

## 🎓 Key Concepts

### 1. **Direct Workflow**
- Everything is in one file
- You write all the methods yourself
- Full control over everything

### 2. **Vectorized Environments**
- 4096 environments run in parallel
- All operations are batched (use tensors)
- Much faster than sequential

### 3. **Reward Shaping**
- Design rewards to guide learning
- Balance between different objectives
- Can be tricky to get right!

### 4. **Observation Design**
- What information does the agent need?
- Cartpole: Need to know pole angle, velocities
- Too little info = can't learn
- Too much info = harder to learn

---

## 🧪 Try It Yourself

### Train Cartpole:
```bash
cd scripts/reinforcement_learning/rsl_rl
python train.py --task=Isaac-Cartpole-Direct-v0 --num_envs=4096 --headless
```

### Play with Trained Policy:
```bash
python play.py --task=Isaac-Cartpole-Direct-v0 --num_envs=32
```

---

## ✅ Summary

**CartpoleEnvCfg:**
- Configuration class with all settings
- Defines environment, simulation, robot, scene, rewards

**CartpoleEnv:**
- Main environment class
- Implements all required methods for RL
- Handles observations, actions, rewards, resets

**Key Methods:**
- `_get_observations()` - What the agent sees
- `_apply_action()` - How actions affect the world
- `_get_rewards()` - How good/bad the current state is
- `_get_dones()` - When episode ends
- `_reset_idx()` - How to start new episodes

**This is the foundation** for understanding more complex environments!








