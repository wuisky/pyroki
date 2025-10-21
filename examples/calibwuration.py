"""Online Planning

Run online planning in collision aware environments.
"""

import time
from pathlib import Path

import numpy as np
import pytorch_kinematics
import torch
import trimesh
import viser
from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer
from PIL import Image
from robot_descriptions.loaders.yourdfpy import (load_robot_description,
                                                 yourdfpy)
from viser.extras import ViserUrdf

import pyroki as pk
from pyroki.collision import HalfSpace, RobotCollision, Sphere


class MaskRenderer:
    def __init__(self, urdf_path: Path):
        # # todo:read camera info from config file
        # width = 640
        # height = 480
        # self.intrinsic = torch.tensor([
        #     [554.25469, 0, width/2],
        #     [0, 554.25469, height/2],
        #     [0, 0, 1]
        # ], dtype=torch.float32).cuda()
        width = 1920
        height = 1080
        self.intrinsic = torch.tensor([
            [1125.966064453125, 0.0, 951.341552734375],
            [ 0.0, 1125.034912109375, 532.13677978515625],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32).cpu().numpy()

        self.urdf = yourdfpy.URDF.load(str(urdf_path))
        self.visible_links = {
            link_name: str(urdf_path.parent/link.visuals[0].geometry.mesh.filename)
            for link_name, link in self.urdf.link_map.items()
            if link.visuals and link.visuals[0].geometry and
            link.visuals[0].geometry.mesh
            # if urdf.link_map[link_name].visuals
        }
        print(f'{self.visible_links=}')
        print(f'{list(self.visible_links.keys())[-1]=}')
        self.chain = pytorch_kinematics.build_serial_chain_from_urdf(
            open(urdf_path, mode='rb').read(),
            list(self.visible_links.keys())[-1])

        self.map_link2verts = {}
        self.map_link2faces = {}
        for link_name, mesh_path in self.visible_links.items():
            print(f'Link: {link_name}, Mesh: {mesh_path}')
            mesh = trimesh.load(mesh_path, force='mesh')
            vertices = torch.from_numpy(mesh.vertices).float().cuda()
            faces = torch.from_numpy(mesh.faces).int().cuda()
            # print(f'{vertices.shape=}, {faces.shape=}')
            self.map_link2verts[link_name] = vertices
            self.map_link2faces[link_name] = faces

        self.renderer = NVDiffrastRenderer(height, width)

    def render(self, q, T_c2b):
        fk_ret = self.chain.forward_kinematics(q, end_only=False)
        for link_name in self.visible_links:
            # print(f'{link=} {fk_ret[link]=}')
            link = self.urdf.link_map[link_name]
            if link.visuals and link.visuals[0].origin is not None:
                T_visual_link = link.visuals[0].origin
            else:
                T_visual_link = np.eye(4)

            T_b2l = (fk_ret[link_name].get_matrix().squeeze(0) @
                     torch.from_numpy(T_visual_link).float())
            T_c2l = T_c2b @ T_b2l
            self.renderer.append_mesh(self.map_link2verts[link_name],
                                      self.map_link2faces[link_name],
                                      T_c2l.cuda(),
                                      self.intrinsic)
        mask = self.renderer.render_mask_batched(anti_aliasing=True)
        self.renderer.clear_mesh()

        return mask.detach().cpu().numpy()


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config

def load_local_image(image_path: Path):
    # ローカルJPG画像を読み込み
    # jpg_image_path = Path(__file__).parent / "raw_image.jpg"
    jpg_pil_image = Image.open(image_path)
    # RGB形式に変換（必要に応じて）
    if jpg_pil_image.mode != 'RGB':
        jpg_pil_image = jpg_pil_image.convert('RGB')

    # 大きな画像はリサイズ（最大640x480）
    max_width, max_height = 640, 480
    if jpg_pil_image.width > max_width or jpg_pil_image.height > max_height:
        jpg_pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

    return jpg_pil_image

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # # todo: pass from command line
    urdf_path = Path(__file__).parent / '../wur5e/ur5e.urdf'
    urdf = yourdfpy.URDF.load(str(urdf_path))
    # target_link_name = "tool0"
    renderer = MaskRenderer(urdf_path)

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # create camera
    mesh = trimesh.load_mesh(Path(__file__).parent / '../mesh/security_camera.stl')
    mesh.apply_scale(0.001)
    obj_handle = server.scene.add_transform_controls(
        "/camera",
        scale=0.2,
        wxyz=(0, 0, 1, 0),
        position=(0.4, 0.0, 0.7)
    )
    # viserで可視化
    server.scene.add_mesh_trimesh("/camera/visual", mesh=mesh, wxyz=(0, 0.707, 0, 0.707))

    # Create sliders in GUI that help us move the robot joints.
    update_joint_btn = server.gui.add_button(
        label='実機の関節角度を取得',
    )
    with server.gui.add_folder("Joint position control",
                               expand_by_default=False):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, urdf_vis
        )
        urdf_vis.update_cfg(initial_config)
        for q, s in zip(initial_config, slider_handles):
            s.value = q

    snapshot_cache = None
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        snapshot_btn = client.gui.add_button(
            label='撮像',  # ボタンに表示されるテキスト
        )
        # 画像更新ボタン
        calib_btn = client.gui.add_button(
                    label='キャリブレーション開始',
                    disabled=True,
        )

        @snapshot_btn.on_click
        def _(_) -> None:
            nonlocal snapshot_cache
            # 撮像開始の通知
            capture_notif = client.add_notification(
                title="撮像中",
                body="画像を取得しています...",
                loading=True,
                with_close_button=False,
            )

            # ローカルJPG画像を読み込み
            jpg_image_path = Path(__file__).parent / "raw_image.jpg"
            jpg_pil_image = load_local_image(jpg_image_path)
            jpg_image_handle.image = np.array(jpg_pil_image)
            calib_btn.disabled = False
            snapshot_cache = np.array(jpg_pil_image)


            # 撮像完了の通知
            capture_notif.remove()
            client.add_notification(
                title="撮像完了",
                body="画像が正常に取得されました",
                auto_close_seconds=5,
            )
            # @modal_button.on_click
            with client.gui.add_modal('この画像でいいの？') as modal:
                # 取得した画像をモーダル内に表示
                modal_img_np = np.array(jpg_pil_image)
                client.gui.add_image(
                    modal_img_np,
                    format="jpeg"
                )

                client.gui.add_markdown('再撮像しない場合、カメラの位置を調整して計算マスク'
                                        'を写真のロボットのシルエット'
                                        'に大体合うようにしてから'
                                        'キャリブレーション開始を押してください')

                def run_sam():
                    modal.close()
                    # SAM処理開始の通知
                    loading_notif = client.add_notification(
                        title="SAM処理中",
                        body="Segment Anything Modelで画像を解析しています...",
                        loading=True,
                        with_close_button=False,
                    )

                    # ここでSAM（Segment Anything Model）の処理を実行
                    # 実際の処理をシミュレートするため少し待機
                    time.sleep(2)  # 実際のSAM処理に置き換える

                    # 処理完了の通知
                    loading_notif.remove()  # ローディング通知を削除

                    client.add_notification(
                        title="SAM処理完了",
                        body="画像の解析が完了しました。ロボットマスク画像確認しろください",
                        loading=False,
                        with_close_button=True,
                        auto_close_seconds=3,  # 3秒後に自動で閉じる
                    )

                client.gui.add_button("この画像でいく").on_click(lambda _: run_sam())
                client.gui.add_button("再撮影する").on_click(lambda _: modal.close())

        @calib_btn.on_click
        def _(_) -> None:
            # キャリブレーション開始の通知
            calib_notif = client.add_notification(
                title="キャリブレーション実行中",
                body="ロボットマスクを計算しています...",
                loading=True,
                with_close_button=False,
            )

            update_mask()

            # キャリブレーション完了の通知
            calib_notif.remove()
            client.add_notification(
                title="キャリブレーション完了",
                body="マスク計算が完了しました。",
                auto_close_seconds=4,
            )


    # PIL画像の表示例
    # 1. NumPy配列からPIL画像を作成
    img_array = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)
    pil_image = Image.fromarray(img_array)

    # 2. PIL画像をnumpy配列に変換してViserで表示（3Dシーン内）
    img_np = np.array(pil_image)
    # server.scene.add_image(
    #     "/pil_image_3d",
    #     image=img_np,
    #     render_width=1.0,
    #     render_height=1.0,
    #     format="png",
    #     position=(0.0, 0.0, 1.0),
    #     wxyz=(1.0, 0.0, 0.0, 0.0)
    # )

    # 3. GUI内での画像表示
    with server.gui.add_folder("Images"):
        # JPG画像表示用のハンドル
        jpg_image_handle = server.gui.add_image(
            img_np,
            label="カメラ画像",
            format="jpeg"
        )

        # マスク画像表示用のハンドル
        mask_image_handle = server.gui.add_image(
            img_np,
            label="計算マスク画像",
            format="png"
        )

    def update_mask():
        T_b2c = np.eye(4)
        T_b2c[:3, 3] = obj_handle.position
        rot = trimesh.transformations.quaternion_matrix(obj_handle.wxyz)
        T_b2c[:3, :3] = rot[:3, :3]
        T_c2b = np.linalg.inv(T_b2c)
        T_c2b = torch.from_numpy(T_c2b).float()
        q = [q.value for q in slider_handles]
        try:
            mask_np = renderer.render(q, T_c2b)
        except Exception as e:
            print(f"レンダリングエラー: {e}")
            return

        # マスク画像を適切な形式に変換
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]  # バッチ次元を削除

        # マスクサイズを制限（640x480）
        max_width, max_height = 640, 480
        if mask_np.shape[0] > max_height or mask_np.shape[1] > max_width:
            # PILを使用してリサイズ
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
            mask_pil.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            mask_np = np.array(mask_pil) / 255.0  # 0-1の範囲に戻す

        # マスクを0-255の範囲に正規化
        mask_normalized = (mask_np * 255).astype(np.uint8)

        # グレースケールマスクをRGB画像に変換
        if mask_normalized.ndim == 2:
            img = np.stack([mask_normalized, mask_normalized, mask_normalized], axis=2)
        else:
            img = mask_normalized
        mask_image_handle.image = img
        base_image = snapshot_cache

        # PILを使ったマスク画像のオーバーレイ
        if base_image is not None and base_image.shape[:2] == img.shape[:2]:
            # NumPy配列をPIL画像に変換
            base_pil = Image.fromarray(base_image).convert("RGBA")
            mask_pil = Image.fromarray(np.mean(img, axis=2).astype(np.uint8))

            # オーバーレイ用の緑色画像を作成
            overlay = Image.new("RGBA", base_pil.size, (255, 0, 0, 0))

            # マスクを半透明に変換（0-255の範囲で128=50%透明度）
            overlay_mask = mask_pil.point(lambda x: 128 if x > 128 else 0)
            overlay.putalpha(overlay_mask)

            # アルファ合成でオーバーレイ
            blended = Image.alpha_composite(base_pil, overlay)
            # RGBに変換してNumPy配列に戻す
            blended_rgb = blended.convert("RGB")
            jpg_image_handle.image = np.array(blended_rgb)

        # print(f"マスク画像サイズ: {img.shape}")


    ######## callback
    @update_joint_btn.on_click
    def _(_) -> None:
        th_deg = [-99.57, -149.865, -61.46, 0.0, 90.0, 0.0]
        th_rad = [np.deg2rad(angle) for angle in th_deg]
        urdf_vis.update_cfg(th_rad)
        # for _, (slider, angle_rad) in enumerate(zip(slider_handles, th_rad)):
        #    slider.value = angle_rad
        for _, (slider, angle_rad) in enumerate(zip(slider_handles, th_rad)):
            slider.value = angle_rad

        # # 新しいランダム画像を生成
        # new_img_array = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)
        # new_pil_image = Image.fromarray(new_img_array)
        # new_img_np = np.array(new_pil_image)

        # # # 3Dシーン内の画像を更新
        # # server.scene.add_image(
        # #     "/pil_image_3d",
        # #     image=new_img_np,
        # #     render_width=1.0,
        # #     render_height=1.0,
        # #     format="png",
        # #     position=(0.0, 0.0, 1.0),
        # #     wxyz=(1.0, 0.0, 0.0, 0.0)
        # # )

        # # GUI内の画像を更新
        # gui_image_handle.image = new_img_np
        # print("画像を更新しました")

    @obj_handle.on_update
    def _(_) -> None:
        # print(f'{obj_handle.position=}, {obj_handle.wxyz=}')
        update_mask()


    print('ready')
    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
