import unittest

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.import_utils import register_third_party_plugins


class PluginDiscoveryTest(unittest.TestCase):
    def test_lerobot_discovers_all_local_plugins(self) -> None:
        register_third_party_plugins()
        self.assertIn("mujoco", RobotConfig.get_known_choices())
        self.assertIn("mocap_ros", TeleoperatorConfig.get_known_choices())


if __name__ == "__main__":
    unittest.main()
