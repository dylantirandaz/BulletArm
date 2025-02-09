import pybullet as p
import numpy as np
import inspect
import traceback
from bulletarm import env_factory

def runDemo():
    p.connect(p.DIRECT)  # Set PyBullet to direct mode

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

            if hasattr(env, "action_space"):
                print("Action space details:", env.action_space)

            print("Step function signature:", inspect.signature(env.step))  
            print("Step function documentation:", env.step.__doc__)  

        else:
            print("ERROR: `env_runner` does not contain an environment instance.")
            return

        obs = env.reset()
        done = False

        while not done:
            # Generate a valid action based on action space
            if hasattr(env, "action_space"):
                action = env.action_space.sample()
            else:
                action = np.array([0.5, -0.2, 3.14])  # Fallback action

            # Ensure action is in the correct format
            if isinstance(action, int):
                action = [action]  # Convert single int into a list
            elif isinstance(action, tuple):
                action = list(action)  # Convert tuple into a list
            elif isinstance(action, dict):
                action = action  # Use as is
            else:
                action = np.array(action)  # Convert to numpy array

            print("Trying action:", action, "Type:", type(action))

            try:
                obs, reward, done = env.step(action)  
            except Exception as e1:
                print("Error in step():", e1)
                traceback.print_exc()

                print("Trying action as dictionary...")
                try:
                    obs, reward, done = env.step({"action": action})
                except Exception as e2:
                    print("Error using dictionary:", e2)
                    traceback.print_exc()

                    print("Trying action as tuple...")
                    try:
                        obs, reward, done = env.step((action,))
                    except Exception as e3:
                        print("Error using tuple:", e3)
                        traceback.print_exc()

    except Exception as e:
        print(f"Execution error: {str(e)}")
    finally:
        if 'env_runner' in locals() and env_runner is not None:
            env_runner.close()

if __name__ == "__main__":
    runDemo()
