import unittest
import numpy as np
from ship_env import ShipClarke83Env

class TestShipClarke83Env(unittest.TestCase):
    def setUp(self):
        self.env = ShipClarke83Env()

    def test_reward_at_target(self):
        # At target position, heading, and speed
        obs = np.array([
            0,   # x
            0,   # y
            0,   # psi
            0,   # u
            0.0,                  # v
            0.0                   # r
        ]) / self.env.norm_scale
        action = np.array([0.0, 0.0])
        reward, terminated = self.env._calculate_reward(obs, action)
        print(f"env_target: {self.env.target}")
        print(f"Reward at target: {reward}, Terminated: {terminated}")
        # self.assertTrue(terminated)
        # self.assertGreaterEqual(reward, 50)

    def test_reward_far_from_target(self):
        # Far from target
        obs = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]) / self.env.norm_scale
        action = np.array([0.0, 0.0])
        reward, terminated = self.env._calculate_reward(obs, action)
        self.assertFalse(terminated)
        self.assertLess(reward, 10)

    def test_reward_with_action_penalty(self):
        # At target but with large action
        obs = np.array([
            self.env.target[0], self.env.target[1], self.env.target[2],
            self.env.target[3], 0.0, 0.0
        ]) / self.env.norm_scale
        action = np.array([10.0, 10.0])
        reward, terminated = self.env._calculate_reward(obs, action)
        self.assertTrue(terminated)
        self.assertLess(reward, 70)  # Should be less than max due to penalty


if __name__ == '__main__':
    unittest.main()