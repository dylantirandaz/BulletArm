import pybullet as p
import numpy as np
from bulletarm import env_factory

def runDemo():
    p.connect(p.DIRECT) 
    env_config = {
        'render': True,
        'render_mode': 'gui'
    }

    try:
        env_runner = env_factory.createSingleProcessEnv(env_type='block_stacking', env_config=env_config)
        print("Environment instance:", env_runner)

        if env_runner is None:
            print("ERROR: `createSingleProcessEnv()` returned None.")
            return

        if hasattr(env_runner, 'env'):
            env = env_runner.env
            print("Actual environment inside env_runner:", env)
            print("Available methods in env:", dir(env))
            print("action_space content:", env.action_space)  

            if hasattr(env, 'action_shape'):
                print("Expected action shape:", env.action_shape)  

            if not hasattr(env, 'num_solver_iterations'):
                env.num_solver_iterations = 150
                print("Manually set num_solver_iterations = 150")

            if not hasattr(env, 'solver_residual_threshold'):
                env.solver_residual_threshold = 1e-5
                print("Manually set solver_residual_threshold = 1e-5")

        else:
            print("ERROR: `env_runner` does not contain an environment instance.")
            return

        obs = env.reset()
        done = False

        while not done:
            action_space_np = np.array(env.action_space)
            action = action_space_np[np.random.choice(action_space_np.shape[0])]

            if action.shape[0] > 2:
                action = action[:2]  
            elif action.shape[0] < 2:
                raise ValueError(f"Selected action is too short: {action}, shape: {action.shape}")

            print("Using action:", action, "Shape:", action.shape)  
            print("Expected action input format:", type(action), action.shape)

            obs, reward, done, _ = env.step(action) 

    except Exception as e:
        print(f"Execution error: {str(e)}")
    finally:
        if 'env_runner' in locals() and env_runner is not None:
            env_runner.close()

if __name__ == "__main__":
    runDemo()
