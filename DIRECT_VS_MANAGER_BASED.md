# Direct vs Manager-Based Workflows in Isaac Lab

## Overview

Isaac Lab provides **two different workflows** for designing RL environments:

1. **Direct Workflow** (`direct/`) - Full control, write everything yourself
2. **Manager-Based Workflow** (`manager_based/`) - Modular, use managers for components

---

## Key Differences

| Aspect | Direct Workflow | Manager-Based Workflow |
|--------|----------------|----------------------|
| **Philosophy** | Write everything in one class | Configure separate managers |
| **Code Location** | Single environment class | Config file + manager classes |
| **Complexity** | Lower-level, more code | Higher-level, less code |
| **Flexibility** | Full control over everything | Modular, easy to swap components |
| **Best For** | Performance optimization, complex logic | Prototyping, collaboration, modularity |
| **Base Class** | `DirectRLEnv` | `ManagerBasedRLEnv` |

---

## Direct Workflow

### Structure

```
direct/cartpole/
├── cartpole_env.py          ← Single file with everything
└── agents/
    └── rsl_rl_ppo_cfg.py
```

### How It Works

You inherit from `DirectRLEnv` and implement **all methods yourself**:

```python
class CartpoleEnv(DirectRLEnv):
    cfg: CartpoleEnvCfg
    
    def _setup_scene(self):
        # You write: Create robot, ground plane, etc.
        self.cartpole = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(...)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        # You write: Process actions before physics
        self.actions = actions.clone()
    
    def _apply_action(self):
        # You write: Apply actions to simulator
        self.cartpole.set_joint_effort_target(
            self.actions * self.cfg.action_scale, 
            joint_ids=self._cart_dof_idx
        )
    
    def _get_observations(self) -> dict:
        # You write: Compute observations manually
        obs = torch.cat((
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
        ), dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        # You write: Compute rewards manually
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            # ... all reward terms computed here
        )
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # You write: Check termination conditions
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos
        return out_of_bounds, time_out
    
    def _reset_idx(self, env_ids):
        # You write: Reset logic
        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        # ... reset everything manually
```

### Characteristics

✅ **Pros:**
- **Full control - write exactly what you want
- **Performance** - No overhead from manager abstraction
- **Fine-grained control** - Every detail is explicit
- **Simple for small tasks** - Everything in one place

❌ **Cons:**
- **More code** - You write everything from scratch
- **Less modular** - Hard to swap components
- **Harder to collaborate** - Everything mixed together
- **More error-prone** - You handle all edge cases

### Example: Cartpole Direct

**File**: `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py`

```python
# Everything is in one class:
class CartpoleEnv(DirectRLEnv):
    def _get_observations(self):
        # Manual observation computation
        obs = torch.cat((...), dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self):
        # Manual reward computation
        return compute_rewards(...)
    
    def _get_dones(self):
        # Manual termination check
        return out_of_bounds, time_out
```

---

## Manager-Based Workflow

### Structure

```
manager_based/locomotion/velocity/config/digit/
├── rough_env_cfg.py         ← Configuration (WHAT)
│   ├── DigitObservations    ← Observation config
│   ├── DigitRewards         ← Reward config
│   ├── ActionsCfg          ← Action config
│   └── TerminationsCfg      ← Termination config
└── agents/
    └── rsl_rl_ppo_cfg.py
```

### How It Works

You define **configuration classes** and managers handle the implementation:

```python
# In rough_env_cfg.py - Just configuration!
@configclass
class DigitObservations:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,  # ← Function to call
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

@configclass
class DigitRewards:
    termination_penalty = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-100.0,
    )
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
    )

@configclass
class DigitRoughEnvCfg(ManagerBasedRLEnvCfg):
    rewards: DigitRewards = DigitRewards()
    observations: DigitObservations = DigitObservations()
    # ... other configs
```

**Behind the scenes**, `ManagerBasedRLEnv` creates managers:

```python
# In ManagerBasedRLEnv.load_managers() (you don't write this!)
self.observation_manager = ObservationManager(self.cfg.observations, self)
self.reward_manager = RewardManager(self.cfg.rewards, self)
self.action_manager = ActionManager(self.cfg.actions, self)
```

**During `env.step()`**, managers use your config:

```python
# ObservationManager.compute() reads your config and calls:
obs = term_cfg.func(self._env)  # Calls mdp.base_lin_vel(env)
obs = term_cfg.noise.func(obs)  # Applies noise

# RewardManager.compute() reads your config and calls:
value = term_cfg.func(self._env) * term_cfg.weight  # Calls mdp.is_terminated(env) * -100.0
```

### Characteristics

✅ **Pros:**
- **Less code** - Just configure, don't implement
- **Modular** - Easy to swap reward/observation components
- **Collaboration-friendly** - Different people work on different managers
- **Consistent** - Managers handle edge cases
- **Reusable** - Same managers work for different tasks

❌ **Cons:**
- **Less control** - Limited by manager capabilities
- **Abstraction overhead** - Slight performance cost
- **Learning curve** - Need to understand manager system
- **Less flexible** - Harder for very custom logic

### Example: Digit Manager-Based

**File**: `source/isaaclab_tasks/.../digit/rough_env_cfg.py`

```python
# Just configuration - no implementation!
@configclass
class DigitObservations:
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, ...)
        # Manager handles the rest

@configclass
class DigitRewards:
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-100.0)
    # Manager handles the rest
```

---

## Side-by-Side Comparison

### Adding a New Observation

#### Direct Workflow:
```python
# In cartpole_env.py
def _get_observations(self) -> dict:
    obs = torch.cat((
        self.joint_pos[:, self._pole_dof_idx[0]],
        self.joint_vel[:, self._pole_dof_idx[0]],
        self.joint_pos[:, self._cart_dof_idx[0]],
        self.joint_vel[:, self._cart_dof_idx[0]],
        # ← You manually add new observation here
        self.cartpole.data.body_pos_w[:, 0, 2],  # New: height
    ), dim=-1)
    return {"policy": obs}
```

#### Manager-Based Workflow:
```python
# In rough_env_cfg.py
@configclass
class DigitObservations:
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, ...)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, ...)
        # ← Just add new config line
        height = ObservationTermCfg(func=mdp.base_height, noise=Unoise(...))
        # Manager automatically handles it!
```

### Adding a New Reward Term

#### Direct Workflow:
```python
# In cartpole_env.py
def _get_rewards(self) -> torch.Tensor:
    rew_alive = self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float())
    rew_termination = self.cfg.rew_scale_terminated * self.reset_terminated.float()
    # ← You manually add new reward term
    rew_height = -0.5 * torch.abs(self.cartpole.data.body_pos_w[:, 0, 2] - 1.0)
    total_reward = rew_alive + rew_termination + rew_height
    return total_reward
```

#### Manager-Based Workflow:
```python
# In rough_env_cfg.py
@configclass
class DigitRewards:
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-100.0)
    track_lin_vel_xy_exp = RewardTermCfg(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0)
    # ← Just add new config line
    height_penalty = RewardTermCfg(func=mdp.base_height_deviation, weight=-0.5)
    # Manager automatically computes and sums it!
```

---

## When to Use Which?

### Use **Direct Workflow** when:
- ✅ You need **maximum performance** (no abstraction overhead)
- ✅ You have **very custom logic** that doesn't fit managers
- ✅ You're building a **simple, one-off task**
- ✅ You want **full control** over every detail
- ✅ You're **optimizing existing code** for speed

**Examples**: `cartpole`, `humanoid`, `ant` (simple tasks)

### Use **Manager-Based Workflow** when:
- ✅ You're **prototyping** and experimenting with different configurations
- ✅ You're working in a **team** (different people work on different managers)
- ✅ You want **modularity** (easy to swap reward functions, observations)
- ✅ You're building **complex tasks** with many components
- ✅ You want **consistency** across different tasks

**Examples**: `locomotion/velocity`, `manipulation/reach`, `locomanipulation` (complex tasks)

---

## Code Comparison Example

### Direct: Cartpole

```python
# Everything in one file: cartpole_env.py
class CartpoleEnv(DirectRLEnv):
    def _get_observations(self):
        # 15+ lines of manual observation computation
        obs = torch.cat((...), dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self):
        # 10+ lines of manual reward computation
        return compute_rewards(...)
    
    def _get_dones(self):
        # 5+ lines of manual termination check
        return out_of_bounds, time_out
```

**Total**: ~150 lines of implementation code

### Manager-Based: Digit

```python
# Configuration file: rough_env_cfg.py
@configclass
class DigitObservations:
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, ...)
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, ...)
        # ... 8 observation terms, just config!

@configclass
class DigitRewards:
    termination_penalty = RewardTermCfg(func=mdp.is_terminated, weight=-100.0)
    track_lin_vel_xy_exp = RewardTermCfg(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0)
    # ... 20+ reward terms, just config!
```

**Total**: ~200 lines of **configuration** (not implementation!)

The managers handle all the implementation automatically.

---

## Architecture Comparison

### Direct Workflow Architecture

```
DirectRLEnv (base class)
    ↓
YourEnv(DirectRLEnv)
    ├── _setup_scene()          ← You implement
    ├── _pre_physics_step()     ← You implement
    ├── _apply_action()         ← You implement
    ├── _get_observations()      ← You implement
    ├── _get_rewards()           ← You implement
    ├── _get_dones()             ← You implement
    └── _reset_idx()             ← You implement
```

**You write everything!**

### Manager-Based Workflow Architecture

```
ManagerBasedRLEnv (base class)
    ├── ObservationManager      ← Generic implementation
    ├── RewardManager           ← Generic implementation
    ├── ActionManager           ← Generic implementation
    ├── TerminationManager      ← Generic implementation
    └── CommandManager          ← Generic implementation
        ↓
YourEnvCfg (configuration)
    ├── observations: YourObservations  ← You configure
    ├── rewards: YourRewards           ← You configure
    ├── actions: YourActions           ← You configure
    └── terminations: YourTerminations ← You configure
```

**You configure, managers implement!**

---

## Summary

| Question | Direct | Manager-Based |
|----------|--------|---------------|
| **Where is the code?** | In your environment class | In config + manager classes |
| **What do you write?** | All methods (`_get_observations`, `_get_rewards`, etc.) | Just configuration classes |
| **How to add observation?** | Modify `_get_observations()` method | Add `ObservationTermCfg` to config |
| **How to add reward?** | Modify `_get_rewards()` method | Add `RewardTermCfg` to config |
| **Performance?** | Faster (no abstraction) | Slightly slower (abstraction) |
| **Modularity?** | Low (everything together) | High (separate managers) |
| **Best for?** | Simple tasks, performance | Complex tasks, collaboration |

---

## Real Examples in Codebase

### Direct Examples:
- `direct/cartpole/` - Simple cartpole task
- `direct/humanoid/` - Humanoid locomotion
- `direct/ant/` - Ant locomotion
- `direct/franka_cabinet/` - Manipulation task

### Manager-Based Examples:
- `manager_based/locomotion/velocity/` - Velocity tracking for many robots
- `manager_based/manipulation/reach/` - Reach tasks
- `manager_based/manipulation/lift/` - Lift tasks
- `manager_based/locomanipulation/` - Combined locomotion + manipulation

---

## Migration Path

You can start with **Manager-Based** for prototyping, then migrate to **Direct** if you need:
- More performance
- Very custom logic
- Fine-grained control

Many tasks in Isaac Lab use Manager-Based because it's easier to maintain and extend!








