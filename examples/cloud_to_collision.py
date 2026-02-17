#!/usr/bin/env python3
"""
cloud.ply を読み込んで点群の統計を出し、可能なら表示します。

依存:
  pip install trimesh pyglet
  # 表示しない/できない環境でも読み込みと統計は動きます
"""

import numpy as np
import trimesh
from pathlib import Path
import open3d as o3d
import pyroki as pk
import viser
import time

from wutility import sample_even_fit_mesh, voxel_fit_volume_sample_surface_mesh

def load_pointcloud(path: str) -> trimesh.points.PointCloud:
    print(f"Loading: {path}")
    geom = trimesh.load(path)  # trimeshが自動判別（PointCloud/Meshなど）
    print(f"Loaded type: {type(geom)}")

    # trimesh.load は Mesh を返すこともあるので吸収
    if isinstance(geom, trimesh.Scene):
        # Scene の場合は全部まとめる（点群/メッシュ混在の可能性）
        geoms = list(geom.geometry.values())
        print(f"Scene with {len(geoms)} geometries")
        if not geoms:
            raise ValueError("PLYの中身が空です")
        geom = trimesh.util.concatenate(geoms)

    if isinstance(geom, trimesh.Trimesh):
        # Mesh として読まれた場合は頂点を点群として扱う
        pc = trimesh.points.PointCloud(geom.vertices, colors=geom.visual.vertex_colors)
        return pc

    if isinstance(geom, trimesh.points.PointCloud):
        return geom

    raise TypeError(f"Unsupported type: {type(geom)}")


def downsample_pointcloud_to_spheres(pc: trimesh.points.PointCloud,
                                     n_spheres: int = None,
                                     voxel_size: float = None,
                                     sphere_radius: float = 0.005) -> pk.collision.Sphere:
    """
    点群を直接ダウンサンプリングしてSpheresに変換

    Parameters
    ----------
    pc : PointCloud
        入力点群
    n_spheres : int, optional
        目標となる球の数。指定するとvoxel_sizeを自動調整
    voxel_size : float, optional
        ボクセルダウンサンプリングのサイズ [m]
        n_spheresが指定されていない場合は必須
    sphere_radius : float
        各球の半径 [m]

    Returns
    -------
    spheres : pk.collision.Sphere
        球の集合
    """
    pts = np.asarray(pc.vertices)
    print(f"\nDownsampling point cloud:")
    print(f"  Original points: {len(pts)}")

    # Open3Dの点群を作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # n_spheresが指定されている場合、voxel_sizeを自動調整
    if n_spheres is not None:
        print(f"  Target spheres: {n_spheres}")

        # 点群のバウンディングボックスから初期voxel_sizeを推定
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_volume = bbox.volume()
        initial_voxel_size = (bbox_volume / n_spheres) ** (1/3)

        # 二分探索でvoxel_sizeを調整
        voxel_low, voxel_high = initial_voxel_size * 0.1, initial_voxel_size * 10
        best_voxel_size = initial_voxel_size

        for _ in range(20):  # 最大20回の反復
            test_voxel = (voxel_low + voxel_high) / 2
            pcd_test = pcd.voxel_down_sample(voxel_size=test_voxel)
            n_test = len(pcd_test.points)

            if abs(n_test - n_spheres) < max(10, n_spheres * 0.05):  # 5%以内または10個以内
                best_voxel_size = test_voxel
                break

            if n_test > n_spheres:
                voxel_low = test_voxel
            else:
                voxel_high = test_voxel

            best_voxel_size = test_voxel

        voxel_size = best_voxel_size
        print(f"  Auto-adjusted voxel size: {voxel_size:.6f} m")

    elif voxel_size is None:
        raise ValueError("Either n_spheres or voxel_size must be specified")

    print(f"  Voxel size: {voxel_size} m")
    print(f"  Sphere radius: {sphere_radius} m")

    # ボクセルダウンサンプリング実行
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts_down = np.asarray(pcd_down.points)

    print(f"  Downsampled points: {len(pts_down)}")
    print(f"  Reduction ratio: {len(pts_down)/len(pts)*100:.1f}%")

    # Spheresに変換
    radii = np.full(len(pts_down), sphere_radius)
    spheres = pk.collision.Sphere.from_center_and_radius(
        center=pts_down, radius=radii)

    # パディング: 足りない分だけradius=0のsphereを追加
    if n_spheres is not None and len(pts_down) < n_spheres:
        n_pad = n_spheres - len(pts_down)
        pad_centers = np.zeros((n_pad, 3))
        pad_radii = np.zeros(n_pad)
        # 既存sphereとpadを連結して新しいSphereを作成
        all_centers = np.concatenate([pts_down, pad_centers], axis=0)
        all_radii = np.concatenate([radii, pad_radii], axis=0)
        spheres = pk.collision.Sphere.from_center_and_radius(center=all_centers, radius=all_radii)
        print(f"  Padded {n_pad} spheres (radius=0)")

    print(f"  ✓ Created {len(spheres.pose.translation())} spheres")
    return spheres


def main():
    cp_path = Path(__file__).parent / "../cad/cloud.ply"
    pc = load_pointcloud(cp_path)

    pts = np.asarray(pc.vertices)
    print(f"  num_points: {len(pts)}")
    if len(pts) > 0:
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = pts.mean(axis=0)
        print(f"  bbox_min: {mins}")
        print(f"  bbox_max: {maxs}")
        print(f"  centroid: {center}")

        has_color = getattr(pc, "colors", None) is not None and len(pc.colors) == len(pc.vertices)
        print(f"  has_color: {has_color}")

        # 点群を直接ダウンサンプリングしてSpheresに変換
        print("\n" + "="*60)
        print("Converting PointCloud to Spheres (Direct Downsampling)")
        print("="*60)

        spheres = downsample_pointcloud_to_spheres(
            pc,
            n_spheres=500,        # 目標球数を指定
            sphere_radius=0.005   # 半径5mm
        )

        # Viserで可視化
        print("\n" + "="*60)
        print("Visualization with Viser")
        print("="*60)
        print("Starting Viser server...")

        server = viser.ViserServer()
        server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        server.gui.configure_theme(dark_mode=True)


        camera_handle = server.scene.add_transform_controls(
            "/camera", scale=0.2,
            wxyz=(0.707, 0.707, 0, 0),
            position=(5.07658497e-01, -7.54795097e-01,  1.37389611e-04),
        )

        # 元の点群を表示（グレー、ダウンサンプリングして軽量化）
        print("Adding original point cloud (downsampled for visualization)...")
        # 表示用に1/10にダウンサンプリング
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(pts)
        pcd_vis_down = pcd_vis.voxel_down_sample(voxel_size=0.01)  # 5mm間隔
        pts_vis = np.asarray(pcd_vis_down.points)

        # 色もダウンサンプリング（最近傍で取得）
        if has_color:
            colors_full = np.asarray(pc.colors)[:, :3]
            # 元の点群から最も近い点の色を使う
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            _, indices = tree.query(pts_vis)
            colors_vis = colors_full[indices]
        else:
            colors_vis = np.full((len(pts_vis), 3), 128, dtype=np.uint8)

        print(f"  Visualization points: {len(pts_vis)} (from {len(pts)})")
        server.scene.add_point_cloud(
            name="/camera/original_pointcloud",
            points=pts_vis,
            colors=colors_vis,
            point_size=0.01  # サイズを大きくして視覚的密度を保つ
        )

        # ダウンサンプリングされたSpheresをメッシュとして表示(赤色)
        print(f"Adding spheres as mesh...")
        sphere_mesh = spheres.to_trimesh()
        server.scene.add_mesh_trimesh(
            name="/camera/coll",
            mesh=sphere_mesh,
        )

        n_spheres = len(spheres.pose.translation())

        print(f"\n✓ Visualization ready!")
        print(f"  Open browser: http://localhost:8080")
        print(f"  Original points: {len(pts)} (gray)")
        print(f"  Spheres: {n_spheres} (red mesh, radius={spheres.radius[0]:.4f}m)")
        print("\nPress Ctrl+C to exit...")

        # サーバーを実行し続ける
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")

    else:
        print('cloud size 0')



if __name__ == "__main__":
    main()
