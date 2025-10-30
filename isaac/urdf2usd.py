from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
from isaacsim.core.api import World

# initialize the world
my_world = World(stage_units_in_meters=1.0)
my_world.reset()
# import omni
# import os
# from isaacsim.asset.importer.urdf import _urdf
# urdf_interface = _urdf.acquire_urdf_interface()

# # setup config params
# import_config = _urdf.ImportConfig()
# import_config.set_merge_fixed_joints(False)
# import_config.set_fix_base(True)

# # parse and import file
# ext_manager = omni.kit.app.get_app().get_extension_manager()
# ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
# extension_path = ext_manager.get_extension_path(ext_id)
# urdf_path = os.path.abspath(
#     extension_path + "/data/urdf/robots/franka_description/robots/"
# )
# filename = "panda_arm_hand.urdf"
# imported_robot = urdf_interface.parse_urdf(urdf_path, filename, import_config)
# urdf_interface.import_robot(urdf_path, filename, imported_robot, import_config, "")
# '/panda'