from bulletarm import env_factory

def runDemo():
    env_config = {
        'render': True,
        'physics_mode': 'direct',
        'render_mode': 'gui'
    }

    env = None
    try:
        env = env_factory.createSingleProcessEnv(env_type='block_stacking', env_config=env_config)

        obs = env.reset()
        done = False
        while not done:
            action = env.getNextAction()
            obs, reward, done, _ = env.step(action)

    except Exception as e:
        print(f"Execution error: {str(e)}")
    finally:
        if env is not None:
            env.close()

if __name__ == "__main__":
    runDemo()
