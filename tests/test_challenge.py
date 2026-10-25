"""Physics/observation regressions; run: python -m unittest discover -s tests -v."""
from pathlib import Path
import sys
import unittest

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dqn"))
from agents.dqn_agent import DQNAgent
from configs.config import AGENT_CONFIG
from environment.challenge_racing import ChallengeCarRacing, speed_penalty
from environment.carracing_env import CarRacingEnv


class ChallengeTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = ChallengeCarRacing(**kwargs)
        self.addCleanup(env.close)
        env.reset(seed=42)
        return env

    def test_classic_matches_original_environment(self):
        env = self.make_env(scenario="classic")
        original = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        self.addCleanup(original.close)
        obs, _ = original.reset(seed=42)
        np.testing.assert_array_equal(env.state, obs)
        for action in (3, 3, 1, 2, 0, 4):
            actual, reward, terminated, truncated, _ = env.step(action)
            expected, expected_reward, t, tr, _ = original.step(action)
            np.testing.assert_array_equal(actual, expected)
            self.assertAlmostEqual(reward, expected_reward)
            self.assertEqual((terminated, truncated), (t, tr))

    def test_scenes_and_obstacles_are_seeded_and_reset_without_leaking_bodies(self):
        for scene in ("forest", "city", "desert"):
            with self.subTest(scene=scene):
                env = self.make_env(scenario=scene)
                layout = [(tuple(o.body.position), o.kind) for o in env.obstacles]
                scenery = env.scenery.copy()
                body_count = env.world.bodyCount
                self.assertGreater(len(layout), 5)
                self.assertGreater(len(scenery), 20)
                self.assertTrue(all(not o.body.fixtures[0].sensor for o in env.obstacles))
                env.reset(seed=42)
                self.assertEqual(layout, [(tuple(o.body.position), o.kind) for o in env.obstacles])
                self.assertEqual(scenery, env.scenery)
                self.assertEqual(body_count, env.world.bodyCount)
                env.reset(seed=123)
                self.assertNotEqual(layout, [(tuple(o.body.position), o.kind) for o in env.obstacles])

    def test_difficulty_increases_hazards_without_changing_track(self):
        normal = self.make_env(difficulty="normal")
        hard = self.make_env(difficulty="hard")
        self.assertEqual(normal.track, hard.track)
        self.assertGreater(len(hard.obstacles), len(normal.obstacles))
        start = np.array(normal.car.hull.position)
        self.assertTrue(all(np.linalg.norm(np.array(o.body.position) - start) > 15 for o in normal.obstacles))

    def test_collision_penalty_and_damage_use_actual_box2d_contacts(self):
        env = self.make_env(difficulty="hard")
        # Bring a solid obstacle to the car to force real Box2D contacts.
        env.obstacles[0].body.position = env.car.hull.position
        _, reward, terminated, _, info = env.step(0)
        self.assertEqual(info["collisions"], 1)
        self.assertEqual(info["damage"], 50)
        self.assertLess(reward, -60)
        self.assertFalse(terminated)
        env.step(0)
        self.assertEqual(env.collisions, 1, "Do not charge each hull fixture/contact frame")
        env.obstacles[1].body.position = env.car.hull.position
        _, _, terminated, _, info = env.step(0)
        self.assertTrue(terminated)
        self.assertEqual(info["end_reason"], "wrecked")
        self.assertEqual(info["damage"], 100)
        env.reset(seed=42)
        self.assertEqual((env.damage, env.collisions), (0, 0))

    def test_obstacles_are_in_agent_pixels_and_display(self):
        env = self.make_env()
        env.t = 1.  # Finish initial camera zoom.
        env.obstacles[0].body.position = env.car.hull.GetWorldPoint((3, 9))
        with_hazards = env._render("state_pixels")
        display = env.render()
        hazards, env.obstacles = env.obstacles, []
        without_hazards = env._render("state_pixels")
        no_display_hazards = env.render()
        env.obstacles = hazards
        self.assertGreater(np.count_nonzero(with_hazards != without_hazards), 10)
        self.assertGreater(np.count_nonzero(display != no_display_hazards), 50)

    def test_rain_reduces_physical_grip(self):
        dry = self.make_env(weather="dry")
        wet = self.make_env(weather="rain")
        self.assertEqual(dry.track, wet.track)
        self.assertTrue(all(t.road_friction == 1. for t in dry.road))
        self.assertTrue(all(t.road_friction < 1. for t in wet.road))
        self.assertTrue(wet.puddles)
        self.assertTrue(any(t.road_friction == .32 for t in wet.road))

    def test_time_limit_preserves_bootstrap_and_observation_contract(self):
        env = CarRacingEnv(dict(scenario="city", max_episode_steps=1))
        self.addCleanup(env.close)
        state = env.reset(seed=42)
        self.assertEqual(state.shape, (4, 84, 84))
        self.assertEqual(state.dtype, np.float32)
        self.assertTrue(np.all((state >= 0) & (state <= 1)))
        _, _, done, info = env.step(3)
        self.assertTrue(done)
        self.assertTrue(info["truncated"])
        self.assertFalse(info["terminated"])
        self.assertEqual(info["end_reason"], "time_limit")

    def test_invalid_scenario_fails_clearly(self):
        with self.assertRaises(ValueError):
            ChallengeCarRacing(scenario="typo")

    def test_speed_penalty_grows_with_speed_and_bend(self):
        self.assertEqual(speed_penalty(150., 0.), 0.)
        self.assertGreater(speed_penalty(250., 0.), 0.)
        self.assertGreater(speed_penalty(150., 1.2), speed_penalty(150., 0.))
        self.assertEqual(speed_penalty(60., 5.), 0.)

    def test_speed_penalty_applies_only_outside_classic(self):
        rewards = {}
        for scenario in ("classic", "forest"):
            env = self.make_env(scenario=scenario)
            env.car.hull.linearVelocity = (0., 80.)  # ~288 km/h, above any safe speed
            rewards[scenario] = env.step(0)[1]
        self.assertLess(rewards["forest"], rewards["classic"])

    def test_replay_memory_roundtrip_is_lossless(self):
        agent = DQNAgent({**AGENT_CONFIG, "memory_size": 4, "batch_size": 1})
        state = (np.random.default_rng(0).integers(0, 256, (4, 84, 84)) / 255.).astype(np.float32)
        agent.remember(state, 1, 0., state, False)
        stored = agent.memory[0][0]
        self.assertEqual(stored.dtype, np.uint8)
        np.testing.assert_array_equal(stored.astype(np.float32) / 255., state)
        self.assertIsNotNone(agent.learn())


if __name__ == "__main__":
    unittest.main()
