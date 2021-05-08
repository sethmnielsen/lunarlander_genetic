import gym
import numpy as np
from collections import deque
from PIL import Image
from multiprocessing import Process, Pipe

class ImageSaver(gym.Wrapper):
    def __init__(self, env, img_path, rank):
        gym.Wrapper.__init__(self, env)
        self._cnt = 0
        self._img_path = img_path
        self._rank = rank

    def step(self, action):
        step_result = self.env.step(action)
        obs, _, _, _ = step_result
        img = Image.fromarray(obs, 'RGB')
        img.save('%s/out%d-%05d.png' % (self._img_path, self._rank, self._cnt))
        self._cnt += 1
        return step_result

def make_env(env_id, img_dir, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        # env.seed(seed + rank)
        if img_dir is not None:
            env = ImageSaver(env, img_dir, rank)
        return env

    return _thunk

# vecenv.py
class VecEnv(object):
    """
    Vectorized environment base class
    """
    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        raise NotImplementedError
    def reset(self):
        """
        Reset all environments
        """
        raise NotImplementedError
    def close(self):
        pass

# subproc_vec_env.py
def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()


    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)

# Create the environment.
def make(env_name, img_dir, num_processes):
    envs = SubprocVecEnv([
        make_env(env_name, img_dir, 1337, i) for i in range(num_processes)
    ])
    return envs
