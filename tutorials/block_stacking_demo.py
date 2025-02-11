import pybullet as p
import numpy as np
import inspect
import traceback
from bulletarm import env_factory

def runDemo():
    #p.connect(p.DIRECT) UNCOMMENT THIS TO RUN WITHOUT BROWSER AND NO GUI 

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
            motion_primitive = np.random.choice([0, 1, 2])
            params = np.random.uniform(low=-1.0, high=1.0, size=(4,))
            action = np.concatenate(([motion_primitive], params))
            
            print("Trying action:", action, "Type:", type(action))
            
            try:
                obs, reward, done = env.step(action)
                print("Step successful. Reward:", reward, "Done:", done)
            except Exception as e:
                print("Error in step() with action:", action)
                traceback.print_exc()
                break

    except Exception as e:
        print(f"Execution error: {str(e)}")
        traceback.print_exc()
    finally:
        if 'env_runner' in locals() and env_runner is not None:
            env_runner.close()

if __name__ == "__main__":
    runDemo()

