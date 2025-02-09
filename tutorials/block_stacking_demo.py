import os
from bulletarm import env_factory

def runDemo():
    env_config = {
        'render': True,
        'physics_mode': 'direct',
        'render_mode': 'gui',
        'robot_model_path': r"C:\Users\dtira_8wmp3o\BulletArm\bulletarm\pybullet\urdf\kuka\kuka_with_gripper2.sdf"
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
            print("Available attributes in actual env:", dir(env))

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
            action = env.getNextAction()
            obs, reward, done, _ = env.step(action)

    except Exception as e:
        print(f"Execution error: {str(e)}")
    finally:
        if 'env_runner' in locals() and env_runner is not None:
            env_runner.close()

if __name__ == "__main__":
    runDemo()
