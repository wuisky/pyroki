from nicegui import ui
from urdf_scene_nicegui.urdf_scene import UrdfScene
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
scene = UrdfScene(os.path.join(script_dir,
                               '../ur5-bullet/UR5/ur_e_description/urdf/ur5e.urdf'))
# scene = UrdfScene('../ur5-bullet/UR5/ur_e_description/urdf/ur5e.urdf')
scene.show(material="#888", scale_stls=1, background_color='#004191')
ui.run()
