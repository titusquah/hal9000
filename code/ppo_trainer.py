import numpy as np
from unstable_reactor_gym import UnstableReactor
from stable_baselines.common.policies import MlpLnLstmPolicy,\
    LstmPolicy,\
    MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines import PPO2
import os


np.seterr(all='raise')  # define before your code.
# %%
nstep = 100
ncpu = 4
n_steps = 0
checkpoints = np.arange(3, 15) ** 6
next_checkpoint = checkpoints[0]
iter_checkpoint = 0


def main():
    global nstep, ncpu, n_steps
    train = True
    tensor_log = False
    for agent in range(1, 2):
        print(agent)
        if tensor_log:
            agent_number = agent
            log_dir = "ppo1/agent{0}".format(agent_number)
            os.makedirs(log_dir, exist_ok=True)
            model_name = 'SS_ppo_model_{0}'.format(agent_number)
        time_steps = int(checkpoints[-1])

        def callback(_locals, _globals):
            """
            Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
            :param _locals: (dict)
            :param _globals: (dict)
            """
            global n_steps, ncpu, nstep, next_checkpoint, iter_checkpoint

            if n_steps * ncpu > next_checkpoint:
                print("Saving new best model")
                _locals['self'].save(
                    log_dir + '/' + model_name + '_{0}_steps'.format(
                        n_steps * ncpu))
                iter_checkpoint += 1
                if iter_checkpoint == 12:
                    n_steps = 0
                    iter_checkpoint = 0
                next_checkpoint = checkpoints[iter_checkpoint]
            #      print(n_steps*ncpu)
            n_steps += 1
            return True

        dt = 1/60/20

        n_cpu = ncpu

        env = SubprocVecEnv([UnstableReactor(dt=dt) for i
                             in range(n_cpu)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True,
                           clip_obs=10.)

        # %%

        if train:
            if tensor_log:

                model = PPO2(MlpLnLstmPolicy, env, verbose=0,
                             tensorboard_log=log_dir, learning_rate=0.00025,
                             nminibatches=n_cpu,
                             n_steps=nstep)
                try:
                    model.learn(total_timesteps=int(time_steps),
                                callback=callback)

                    model.save(
                        log_dir + '/' + model_name + "_{0}_steps".format(
                            time_steps))
                    env.save(log_dir)

                except Exception as e:
                    raise (e)

            else:
                model = PPO2(MlpLnLstmPolicy, env, verbose=0,
                             nminibatches=n_cpu, learning_rate=0.00025,
                             n_steps=nstep)
                model.learn(total_timesteps=int(time_steps))


if __name__ == '__main__':
    main()