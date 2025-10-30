from isaacsim import SimulationApp
import os
import os.path as osp

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension
from isaacsim.asset.importer.urdf import _urdf  # 如果是 4.5+ 版本
import omni.kit.commands

enable_extension("isaacsim.asset.importer.urdf")
enable_extension("omni.kit.widget.stage")
enable_extension("omni.kit.widget.viewport")

# 设置场景：例如添加地面
from isaacsim.core.api import World
world = World()
world.scene.add_default_ground_plane()

# URDF 导入接口
urdf_interface = _urdf.acquire_urdf_interface()

# 配置导入选项
import_config = _urdf.ImportConfig()
import_config.fix_base = True
import_config.merge_fixed_joints = False
import_config.convex_decomp = False
# … 根据需要设置更多参数

urdf_dir = "/home/hhhar/liuliu/vld/assets/articulation"
dest_path = "/World/Robot"  # 在 Stage 中机器人将被放在的 prim 路径

# 执行导入
for data in os.listdir(urdf_dir):
    urdf_path = osp.join(urdf_dir, data, "mobility_annotation_gapartnet.urdf")
    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
            dest_path=dest_path
        )