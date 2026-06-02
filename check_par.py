from Environment import ParallelEnvironment
from torchrl.envs import ParallelEnv,check_env_specs

def main():
    env = ParallelEnv(2,[lambda :ParallelEnvironment('1'), lambda:ParallelEnvironment('2')])
    check_env_specs(env)
    td = env.reset()

    trajectory = []
    #print(env._step())
    print(env.reset())
    print(env.rand_step())

    for _ in range(10):
        td["action"] = env.action_spec.rand()
        td = env.step(td)
        trajectory.append(td)

    print(len(trajectory))
    print(trajectory[0]["observation"].shape)


if __name__ == "__main__":
    main()