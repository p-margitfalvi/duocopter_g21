import numpy as np
import matplotlib.pyplot as plt
from ray.tune.logger import pretty_print
import ray
import gym
import wandb

class SimEnv(gym.Env):

    def __init__(self, env_config):

        self.copter_weight_kg = env_config['copter_weight_kg']
        self.g = env_config['g']
        self.max_thrust_N = env_config['max_thrust_N']
        self.max_wattage_W = env_config['max_wattage_W']
        self.k1_m = env_config['k1_m']
        self.k2_m = env_config['k2_m']
        self.theta_deg = env_config['theta_deg']
        self.dyn_fric_coeff = env_config['dyn_fric_coeff']
        self.cart_height_m = env_config['cart_height_m']
        self.thrust_centerline_dist_m =\
        env_config['thrust_centerline_distance_m']


        self.dt = env_config['dt']
        self.max_height_m = env_config['max_height_m']
        self.sampling_rate_hz = env_config['sampling_rate_hz']
        self.log = env_config['log']

        min_vel = -self.g*np.sqrt(2*self.max_height_m/self.g)*2
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, min_vel,
            -np.inf]),
                high=np.array([self.max_height_m, self.max_height_m,
                    -10*min_vel, np.inf]),
                shape=(4,))

        # Number of simulation steps for one RL step
        self.step_time = int(1/(self.sampling_rate_hz*self.dt))
        self.reset()

    def step(self, action):

        err = 0
        power_consumed_J = 0
        rw = 0
        for sim_step in range(self.step_time):

            if self.t > self.waypoints[0, self.cur_wpt_idx]:
                self.cur_wpt_idx += 1
                if self.cur_wpt_idx == self.waypoints.shape[1]:
                    self.cur_wpt_idx -= 1
                    return self.get_state(),\
                -err*self.dt/(self.step_time*self.waypoints[0, -1]), True, {}

            if self.log:
                self.logs['t'].append(self.t)
                self.logs['y'].append(self.y)
                self.logs['v'].append(self.v)
                self.logs['thrust'].append(action[0])

            self.y += self.v*self.dt
            self.v += self.accel_funciton(action[0], self.y, self.v)*self.dt
            self.t += self.dt

            if self.y < 0:
                self.y = 0
                self.v = max(self.v, 0)

            if self.y > self.max_height_m:
                self.y = self.max_height_m
                self.v = min(self.v, 0)
            tmp = (self.y - self.waypoints[1, self.cur_wpt_idx])*self.dt
            err += np.abs(tmp)
            self.err_accum += tmp
            power_consumed_J += action[0]*self.max_wattage_W*self.dt
            rw -= (0.4/0.7*err + 0.3/0.7*power_consumed_J)

        return self.get_state(), -err/(self.step_time*self.waypoints[0, -1]), False, {}

    def accel_funciton(self, thrust, y, v):
        thrust_term = self.max_thrust_N*thrust
        cable_weight = -0.19*(y + 0.5648)*self.g
        grav_term = -self.copter_weight_kg*self.g + cable_weight

        total_weight_moment = grav_term*self.k1_m + cable_weight*self.k2_m;

        if v == 0:
            resultant = thrust_term + grav_term
            static_friction = 0.13*self.g
            if abs(resultant) < static_friction:
                return 0
            else:
                friction_term = -np.sign(resultant)*static_friction
        else:
            mu = self.dyn_fric_coeff
            l = self.thrust_centerline_dist_m
            theta_rad = self.theta_deg*np.pi/180

            ff1 =\
            abs(mu*(thrust_term*l*np.cos(theta_rad)\
                + total_weight_moment)/self.cart_height_m)
            ff2 =\
            abs(mu*(thrust_term*(self.cart_height_m*np.tan(theta_rad)\
                - l)*np.cos(theta_rad) - total_weight_moment)/self.cart_height_m)

            friction_term = -(ff1 + ff2)*np.sign(v)

        return (thrust_term + grav_term + friction_term) /\
                (self.copter_weight_kg - cable_weight/self.g)

    def reset(self):
        self.y = 0
        self.v = 0
        self.t = 0
        self.err_accum = 0

        self.power_consumed_J = 0

        self.prev_y = 0
        self.prev_v = 0

        self.waypoints = self.generate_waypoints()
        self.cur_wpt_idx = 0

        self.logs = {'t':[], 'y': [], 'v': [], 'thrust': [], 'waypoints':
                self.waypoints}

        return self.get_state()

    def get_state(self):
        # TODO: Add other derivatives
        return [self.waypoints[1, self.cur_wpt_idx], self.y, self.v,\
                self.err_accum]

    def generate_waypoints(self):
        n = np.random.randint(5, 15) # Number of waypoints

        rand_height = lambda low: (self.max_height_m - low) * np.random.rand() + low

        result = np.zeros((2, n))
        result[:, 0] = [3, rand_height(0)]
        for i in range(1, n):
            result[:, i] = (result[0, i-1] + np.random.randint(2, 5), rand_height(0))

        return result

    def plot(self):
        fig, axs = plt.subplots(3, 1, sharex=True)

        # breakpoint()
        axs[0].plot(self.logs['t'], self.logs['y'], label='y')
        axs[0].scatter(self.waypoints[0, :], self.waypoints[1, :], marker='x',
                s=80)
        axs[0].grid()

        axs[1].plot(self.logs['t'], self.logs['v'], label='v')
        axs[1].grid()

        axs[2].plot(self.logs['t'], self.logs['thrust'], label='thrust')
        axs[2].grid()

        plt.show()

    def calc_rms(self):
        # breakpoint()
        time_indices = np.argmin(np.abs(self.logs['t'] - self.waypoints[0, :][np.newaxis].T), 1)
        prev_idx = 0
        err = 0

        for idx_num, idx in enumerate(time_indices):
            err += np.sum(np.square(self.logs['y'][prev_idx:idx] - self.waypoints[1,
                idx_num]))
            prev_idx = idx

        return np.sqrt(err/len(self.logs['t']))


def main_ppo():

    import ray.rllib.agents.ppo as ppo
    wandb.init(project='duocopter', sync_tensorboard=True)
    ray.init()

    env_config = {
        'copter_weight_kg': 0.5,
        'g': 9.81,
        'max_thrust_N': 2*9.81,
        'max_wattage_W': 2*350, # TODO: Use power curve
        'k1_m': 0.01, # TODO: Change
        'k2_m': 24E-3,
        'theta_deg': 0,
        'dyn_fric_coeff': 0.14,
        'cart_height_m': 0.2104,
        'thrust_centerline_distance_m': 0.01, #TODO: Change
        'dt': 5E-4,
        'max_height_m': 1.44,
        'sampling_rate_hz': 20,
        'log': True
        }

    config = ppo.DEFAULT_CONFIG.copy()

    config['num_workers'] = 10
    config['env_config'] = env_config
    config['lambda'] = 0.9
    config['lr'] = 5e-5
    config['rollout_fragment_length'] = 500
    config['model']['fcnet_hiddens'] = [64, 64]

    trainer = ppo.PPOTrainer(config=config, env=SimEnv)

    for i in range(300):
        result = trainer.train()
        print(pretty_print(result))

    env = SimEnv(env_config)
    state = env.reset()
    done = False
    ep_reward = 0

    while not done:
        thrust = trainer.compute_action(state, explore=False)
        state, rw, done, _ = env.step(thrust)
        ep_reward += rw

    print(env.calc_rms())
    env.plot()
    checkpoint = trainer.save()
    print(checkpoint)

def main_sac():

    from ray.rllib.agents.sac import SACTrainer, DEFAULT_CONFIG

    wandb.init(project='duocopter', sync_tensorboard=True)
    ray.init()

    env_config = {
        'copter_weight_kg': 0.5,
        'g': 9.81,
        'max_thrust_N': 2*9.81,
        'max_wattage_W': 2*350, # TODO: Use power curve
        'k1_m': 0.01, # TODO: Change
        'k2_m': 24E-3,
        'theta_deg': 0,
        'dyn_fric_coeff': 0.14,
        'cart_height_m': 0.2104,
        'thrust_centerline_distance_m': 0.01, #TODO: Change
        'dt': 1E-3,
        'max_height_m': 1.44,
        'sampling_rate_hz': 20,
        'log': True
        }

    config = DEFAULT_CONFIG.copy()

    config['num_workers'] = 10
    config['env_config'] = env_config
    config['framework'] = 'torch'
    config['Q_model']['fcnet_hiddens'] = [64, 64]
    config['policy_model']['fcnet_hiddens'] = [64, 64]
    config['timesteps_per_iteration'] = 5000
    config['rollout_fragment_length'] = 1
    config['buffer_size'] = 30000
    config['prioritized_replay'] = True
    config['train_batch_size'] = 1024
    config['n_step'] = 5
    config['target_network_update_freq'] = 5
    #config['lambda'] = 0.9
    #config['lr'] = 5e-5
    #config['rollout_fragment_length'] = 500
    #config['model']['fcnet_hiddens'] = [64, 64]

    trainer = SACTrainer(config=config, env=SimEnv)

    for i in range(100):
        result = trainer.train()
        print(pretty_print(result))

    env = SimEnv(env_config)
    state = env.reset()
    done = False
    ep_reward = 0

    while not done:
        thrust = trainer.compute_action(state, explore=False)
        state, rw, done, _ = env.step(thrust)
        ep_reward += rw

    print(env.calc_rms())
    env.plot()

def test():
    env_config = {
        'copter_weight_kg': 0.5,
        'g': 9.81,
        'max_thrust_N': 2*9.81,
        'max_wattage_W': 2*350, # TODO: Use power curve
        'k1_m': 0.01, # TODO: Change
        'k2_m': 24E-3,
        'theta_deg': 0,
        'dyn_fric_coeff': 0.14,
        'cart_height_m': 0.2104,
        'thrust_centerline_distance_m': 0.01, #TODO: Change
        'dt': 5E-4,
        'max_height_m': 1.44,
        'sampling_rate_hz': 20,
        'log': True
        }

    env = SimEnv(env_config)
    state = env.reset()
    done = False
    ep_reward = 0

    while not done:
        thrust = [np.random.rand()]
        state, rw, done, _ = env.step(thrust)
        ep_reward += rw

    print(env.calc_rms())
    env.plot()

if __name__ == '__main__':
   # test()
    main_ppo()
