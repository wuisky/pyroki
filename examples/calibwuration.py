"""Online Planning

Run online planning in collision aware environments.
"""

import subprocess
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytorch_kinematics
import torch
import trimesh
import viser
from easyhec.optim.nvdiffrast_cameara_calibrator import (
    CameraInfo,
    NvdiffrastCameraCalibrator,
)
from easyhec.optim.nvdiffrast_renderer import NVDiffrastRenderer
from PIL import Image
from viser.extras import ViserUrdf

from robot_descriptions.loaders.yourdfpy import yourdfpy


class MaskRenderer:
    def __init__(self, urdf_path: Path):
        self.intrinsic = None
        self.renderer = None

        self.urdf = yourdfpy.URDF.load(str(urdf_path))
        self.visible_links = {
            link_name: str(urdf_path.parent/link.visuals[0].geometry.mesh.filename)
            for link_name, link in self.urdf.link_map.items()
            if link.visuals and link.visuals[0].geometry and
            link.visuals[0].geometry.mesh
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
            self.map_link2verts[link_name] = vertices
            self.map_link2faces[link_name] = faces

    def set_camera_info(self, camera_info: CameraInfo):
        self.intrinsic = np.array([
            [camera_info.fx, 0, camera_info.cx],
            [0, camera_info.fy, camera_info.cy],
            [0, 0, 1],
        ], dtype=np.float32)
        self.renderer = NVDiffrastRenderer(camera_info.height, camera_info.width)

    def render(self, q, T_c2b, anti_aliasing=False):
        if self.renderer is None:
            raise RuntimeError('Camera info is not set yet.')
        fk_ret = self.chain.forward_kinematics(q, end_only=False)
        self.renderer.clear_mesh()
        for link_name in self.visible_links:
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
        mask = self.renderer.render_mask_batched(anti_aliasing)
        self.renderer.clear_mesh()
        return mask.detach().cpu().numpy()


class ImageProcessor:
    @staticmethod
    def load_rgb_image(image_path: Path, resize=True):
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        if resize:
            max_width, max_height = 640, 480
            if pil_image.width > max_width or pil_image.height > max_height:
                pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        return pil_image

    @staticmethod
    def load_gray_image(image_path):
        """PIL Imageを使ってマスク画像を読み込み、バイナリ化する"""
        mask_pil = Image.open(image_path).convert('L')
        mask_img = np.array(mask_pil) / 255.0
        mask_bin = mask_img > 0.5
        return mask_bin

    @staticmethod
    def create_overlay(base_image: np.ndarray, mask_image: np.ndarray) -> np.ndarray:
        """ベース画像にマスクをオーバーレイ"""
        if base_image.shape[:2] != mask_image.shape[:2]:
            return base_image

        base_pil = Image.fromarray(base_image).convert('RGBA')
        mask_pil = Image.fromarray(np.mean(mask_image, axis=2).astype(np.uint8))

        overlay = Image.new('RGBA', base_pil.size, (255, 0, 0, 0))
        overlay_mask = mask_pil.point(lambda x: 128 if x > 0 else 0)
        overlay.putalpha(overlay_mask)

        blended = Image.alpha_composite(base_pil, overlay)
        blended_rgb = blended.convert('RGB')
        return np.array(blended_rgb)


class SAMProcessor:
    @staticmethod
    def run_sam_process():
        base_path = '/home/ubuntu/src/Grounded-Segment-Anything'
        sam_command = [
            f'{base_path}/.venv/bin/python',
            f'{base_path}/grounded_sam_demo.py',
            '--config',
            f'{base_path}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
            '--grounded_checkpoint',
            f'{base_path}/groundingdino_swint_ogc.pth',
            '--sam_checkpoint',
            f'{base_path}/sam_vit_h_4b8939.pth',
            '--input_image', '/tmp/snapshot.png',
            '--output_dir', '/tmp/outputs',
            '--box_threshold', '0.3',
            '--text_threshold', '0.25',
            '--text_prompt', 'robot arm',
            '--device', 'cpu'
        ]

        try:
            print('SAM処理を開始します...')
            subprocess.run(
                sam_command,
                capture_output=True,
                text=True,
                timeout=600,
                check=True
            )
            return True
        except Exception as e:
            print(f'SAM処理に予期しないエラー: {e}')
            return False


class CalibrationApp:
    def __init__(self, urdf_path: Path):
        self.urdf_path = urdf_path
        self.urdf = yourdfpy.URDF.load(str(urdf_path))
        self.renderer = MaskRenderer(urdf_path)

        # State variables
        self.snapshot_cache: Optional[np.ndarray] = None
        self.camera_info: Optional[CameraInfo] = None
        self.joint_angles: Optional[List[float]] = None

        # GUI handles
        self.jpg_image_handle = None
        self.mask_image_handle = None
        self.slider_handles: List[viser.GuiInputHandle[float]] = []
        self.obj_handle = None

        # Setup visualizer
        self.server = viser.ViserServer()
        self.setup_scene()
        self.setup_gui()

    def setup_scene(self):
        """3Dシーンのセットアップ"""
        self.server.scene.add_grid('/ground', width=2, height=2, cell_size=0.1)
        self.urdf_vis = ViserUrdf(self.server, self.urdf, root_node_name='/robot')

        # カメラメッシュの設定
        camera_mesh_path = Path(__file__).parent / '../robot_descriptions/security_camera.stl'
        mesh = trimesh.load_mesh(camera_mesh_path)
        mesh.apply_scale(0.003)
        self.obj_handle = self.server.scene.add_transform_controls(
            '/camera',
            scale=0.2,
            wxyz=(0, 0, 1, 0),
            position=(0.4, 0.0, 0.7),
            visible=False,
        )
        self.server.scene.add_mesh_trimesh('/camera/visual', mesh=mesh, wxyz=(0, 0.707, 0, 0.707))

    def setup_gui(self):
        """GUIのセットアップ"""
        # 関節角度取得ボタン
        update_joint_btn = self.server.gui.add_button(label='実機の関節角度を取得')
        update_joint_btn.on_click(self._on_update_joint_angles)

        # スライダーの作成
        with self.server.gui.add_folder('Joint position control', expand_by_default=False):
            self.slider_handles, initial_config = self._create_robot_control_sliders()
            self.urdf_vis.update_cfg(initial_config)
            for q, s in zip(initial_config, self.slider_handles):
                s.value = q

        # 画像表示エリア
        img_array = np.random.randint(0, 256, size=(200, 200, 3), dtype=np.uint8)
        with self.server.gui.add_folder('Images'):
            self.jpg_image_handle = self.server.gui.add_image(
                img_array, label='カメラ画像', format='jpeg'
            )
            self.mask_image_handle = self.server.gui.add_image(
                img_array, label='SAMマスク', format='png'
            )

        # クライアント接続時の処理
        self.server.on_client_connect(self._on_client_connect)

        # オブジェクト更新時の処理
        self.obj_handle.on_update(self._on_camera_update)

    def _create_robot_control_sliders(self):
        """ロボット制御スライダーを作成"""
        slider_handles = []
        initial_config = []

        for joint_name, (lower, upper) in self.urdf_vis.get_actuated_joint_limits().items():
            lower = lower if lower is not None else -np.pi
            upper = upper if upper is not None else np.pi
            initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0

            slider = self.server.gui.add_slider(
                label=joint_name,
                min=lower,
                max=upper,
                step=1e-3,
                initial_value=initial_pos,
                disabled=True,
            )
            slider.on_update(lambda _: self.urdf_vis.update_cfg(
                np.array([s.value for s in slider_handles])
            ))
            slider_handles.append(slider)
            initial_config.append(initial_pos)

        return slider_handles, initial_config

    def _on_client_connect(self, client: viser.ClientHandle):
        """クライアント接続時の処理"""
        snapshot_btn = client.gui.add_button(label='撮像', disabled=True)
        calib_btn = client.gui.add_button(label='キャリブレーション開始', disabled=True)

        # クライアント固有のボタンハンドルを保存
        client.snapshot_btn = snapshot_btn
        client.calib_btn = calib_btn

        snapshot_btn.on_click(lambda _: self._on_snapshot(client, calib_btn))
        calib_btn.on_click(lambda _: self._on_calibration(client))

    def _on_update_joint_angles(self, _):
        """関節角度更新処理"""
        th_deg = [-99.57, -149.865, -61.46, 0.0, 90.0, 0.0]
        self.joint_angles = [np.deg2rad(angle) for angle in th_deg]
        self.urdf_vis.update_cfg(self.joint_angles)

        for slider, angle_rad in zip(self.slider_handles, self.joint_angles):
            slider.value = angle_rad

        # 撮像ボタンを有効化
        for client_id in self.server.get_clients():
            client = self.server.get_clients()[client_id]
            if hasattr(client, 'snapshot_btn'):
                client.snapshot_btn.disabled = False

    def _on_snapshot(self, client: viser.ClientHandle, calib_btn):
        """撮像処理"""
        if self.camera_info is None:
            self._setup_camera_info()

        # 撮像通知
        capture_notif = client.add_notification(
            title='撮像中', body='画像を取得しています...',
            loading=True, with_close_button=False
        )

        # 画像読み込み TODO:PF API
        jpg_image_path = Path(__file__).parent / 'raw_image.jpg'
        jpg_pil_image = ImageProcessor.load_rgb_image(jpg_image_path)
        self.jpg_image_handle.image = np.array(jpg_pil_image)
        self.snapshot_cache = np.array(jpg_pil_image)

        capture_notif.remove()
        client.add_notification(title='撮像完了', body='画像が正常に取得されました', auto_close_seconds=5)

        # モーダル表示
        self._show_confirmation_modal(client, jpg_pil_image, jpg_image_path, calib_btn)

    def _show_confirmation_modal(self, client: viser.ClientHandle, jpg_pil_image: Image.Image,
                                jpg_image_path: Path, calib_btn):
        """確認モーダルを表示"""
        with client.gui.add_modal('この画像でいいの？') as modal:
            modal_img_np = np.array(jpg_pil_image)
            client.gui.add_image(modal_img_np, format='jpeg')
            client.gui.add_markdown('再撮像もできるよ')

            def run_sam():
                modal.close()
                self._run_sam_processing(client, jpg_image_path, calib_btn)

            client.gui.add_button('この画像でいく').on_click(lambda _: run_sam())
            client.gui.add_button('再撮像する').on_click(lambda _: modal.close())

    def _run_sam_processing(self, client: viser.ClientHandle, jpg_image_path: Path, calib_btn):
        """SAM処理を実行"""
        loading_notif = client.add_notification(
            title='SAM処理中', body='Segmentation Anythingでロボット領域検出中',
            loading=True, with_close_button=False
        )

        # NumPy配列からPIL Imageに変換
        snapshot_pil = Image.fromarray(self.snapshot_cache)

        # RGB形式に変換（必要に応じて）
        if snapshot_pil.mode != 'RGB':
            snapshot_pil = snapshot_pil.convert('RGB')

        snapshot_save_path = Path('/tmp/snapshot.png')
        snapshot_pil.save(snapshot_save_path)
        print(f'スナップショットキャッシュを保存しました: {snapshot_save_path}')
        print(f'保存画像サイズ: {snapshot_pil.size}')


        # SAM処理は省略（up_sam_process()）
        sam_image_path = '/tmp/outputs/mask_resize.png'
        if Path(sam_image_path).exists():
            sam_pil_image = ImageProcessor.load_rgb_image(sam_image_path)
            self.mask_image_handle.image = np.array(sam_pil_image)

        loading_notif.remove()
        client.add_notification(
            title='SAM処理完了',
            body='画像の解析が完了しました。カメラの位置を調整してキャリブレーション開始を押してください。'
        )
        calib_btn.disabled = False
        self.obj_handle.visible = True

    def _on_calibration(self, client: viser.ClientHandle):
        """キャリブレーション処理"""
        self.obj_handle.visible = False
        calib_notif = client.add_notification(
            title='キャリブレーション実行中', body='キャリブレーション計算しています...',
            loading=True, with_close_button=False
        )

        try:
            mask_path = '/tmp/outputs/mask_resize.png'
            mask_bin = ImageProcessor.load_gray_image(mask_path)
            robot_masks = np.stack([mask_bin])

            # カメラ姿勢の計算
            T_b2c = self._get_camera_transform()
            T_c2b = np.linalg.inv(T_b2c)

            # キャリブレーション実行
            calibrator = NvdiffrastCameraCalibrator(
                camera_info=self.camera_info,
                urdf_path=self.urdf_path,
            )

            T_c2b_result = calibrator.calibrate(
                q=self.joint_angles,
                robot_masks=robot_masks,
                initial_extrinsic_guess=T_c2b,
            )
            T_b2c_result = np.linalg.inv(T_c2b_result)

            # 結果をオブジェクトに反映
            self._update_camera_from_transform(T_b2c_result)
            self._update_mask()

            calib_notif.remove()
            client.add_notification(
                title='キャリブレーション完了',
                body='キャリブレーション計算が完了しました。赤いシルエットがロボットにピッタリ！'
            )

        except Exception as e:
            calib_notif.remove()
            client.add_notification(
                title='キャリブレーションエラー',
                body=f'エラーが発生しました: {str(e)}'
            )

    def _on_camera_update(self, _):
        '''カメラ更新時の処理'''
        self._update_mask()

    def _setup_camera_info(self):
        """カメラ情報のセットアップ"""
        print('getting camera info from PF API...')
        self.camera_info = CameraInfo(
            width=1920, height=1080,
            fx=1125.966064453125, fy=1125.034912109375,
            cx=951.341552734375, cy=532.13677978515625,
        )

        # 640x480にリサイズ
        target_width, target_height = 640, 480
        scale_x = target_width / self.camera_info.width
        scale_y = target_height / self.camera_info.height
        scale = min(scale_x, scale_y)
        resized_width = int(self.camera_info.width * scale)
        resized_height = int(self.camera_info.height * scale)

        self.camera_info_low = CameraInfo(
            width=resized_width, height=resized_height,
            fx=self.camera_info.fx * scale, fy=self.camera_info.fy * scale,
            cx=self.camera_info.cx * scale, cy=self.camera_info.cy * scale,
        )
        self.renderer.set_camera_info(self.camera_info_low)

    def _get_camera_transform(self) -> np.ndarray:
        """現在のカメラ変換行列を取得"""
        T_b2c = np.eye(4)
        T_b2c[:3, 3] = self.obj_handle.position
        rot = trimesh.transformations.quaternion_matrix(self.obj_handle.wxyz)
        T_b2c[:3, :3] = rot[:3, :3]
        return T_b2c

    def _update_camera_from_transform(self, T_b2c_result: np.ndarray):
        """変換行列からカメラ姿勢を更新"""
        self.obj_handle.visible = True
        pos_b2c = T_b2c_result[:3, 3]
        wxyz_b2c = trimesh.transformations.quaternion_from_matrix(T_b2c_result[:3, :3])
        self.obj_handle.position = pos_b2c
        self.obj_handle.wxyz = wxyz_b2c

    def _update_mask(self):
        """マスク画像を更新"""
        try:
            T_b2c = self._get_camera_transform()
            T_c2b = np.linalg.inv(T_b2c)
            T_c2b = torch.from_numpy(T_c2b).float()
            q = [s.value for s in self.slider_handles]

            mask_np = self.renderer.render(q, T_c2b)

            # マスクサイズを制限（640x480）
            max_width, max_height = 640, 480
            if mask_np.shape[0] > max_height or mask_np.shape[1] > max_width:
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_pil.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                mask_np = np.array(mask_pil) / 255.0

            # マスクを0-255の範囲に正規化
            mask_normalized = (mask_np * 255).astype(np.uint8)

            # グレースケールマスクをRGB画像に変換
            if mask_normalized.ndim == 2:
                img = np.stack([mask_normalized, mask_normalized, mask_normalized], axis=2)
            else:
                img = mask_normalized

            # オーバーレイ処理
            if self.snapshot_cache is not None:
                overlaid_image = ImageProcessor.create_overlay(self.snapshot_cache.copy(), img)
                self.jpg_image_handle.image = overlaid_image

        except Exception as e:
            print(f'マスク更新エラー: {e}')

    def run(self):
        """アプリケーションの実行"""
        print('ready')
        while True:
            time.sleep(10)


def main():
    urdf_path = Path(__file__).parent / '../robot_descriptions/ur5e/ur5e.urdf'
    app = CalibrationApp(urdf_path)
    app.run()


if __name__ == '__main__':
    main()
