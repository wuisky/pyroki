from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, cast

import numpy as onp
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import trimesh
import yourdfpy
from jaxtyping import Array, Float, Int
from loguru import logger

if TYPE_CHECKING:
    from pyroki._robot import Robot

from .._robot_urdf_parser import RobotURDFParser
from ._collision import collide, pairwise_collide
from ._geometry import Capsule, CollGeom, Sphere


@jdc.pytree_dataclass
class RobotCollision:
    """Collision model for a robot, integrated with pyroki kinematics."""

    num_links: jdc.Static[int]
    """Number of links in the model (matches kinematics links)."""
    link_names: jdc.Static[tuple[str, ...]]
    """Names of the links corresponding to link indices."""
    coll: CollGeom
    """Collision geometries for the robot (relative to their parent link frame)."""

    active_idx_i: Int[Array, " P"]
    """Row indices (first link) of active self-collision pairs to check."""
    active_idx_j: Int[Array, " P"]
    """Column indices (second link) of active self-collision pairs to check."""

    num_spheres_per_link: jdc.Static[tuple[int, ...]] = None
    """Number of actual spheres per link (excluding padding). None for capsule-based models."""

    parent_link_indices: jdc.Static[tuple[int, ...]] = None
    """Parent link index for each link. None for root links or models without attachment tracking."""

    def attach_link(
        self,
        new_link_name: str,
        parent_link_name: str,
        spheres: Sphere,
        ignore_self_collision: bool = False,
        offset: jaxlie.SE3 | None = None,
    ) -> "RobotCollision":
        """
        Attach a new link with collision spheres to an existing parent link.

        Args:
            new_link_name: Name for the new link to create.
            parent_link_name: Name of the existing link to attach to.
            spheres: Sphere collision geometry for the new link (in parent link's local frame).
            ignore_self_collision: If True, ignore collisions between new link and parent.
            offset: SE3 transformation offset from parent link frame to new link frame.
                    If None, spheres are used as-is in parent's local frame.

        Returns:
            New RobotCollision instance with the attached link.
        """
        if new_link_name in self.link_names:
            raise ValueError(f"Link '{new_link_name}' already exists in robot collision model")

        if parent_link_name not in self.link_names:
            raise ValueError(f"Parent link '{parent_link_name}' not found in robot collision model")

        parent_idx = self.link_names.index(parent_link_name)

        # Get the number of spheres to attach
        attach_batch_axes = spheres.get_batch_axes()
        if len(attach_batch_axes) == 0:
            # Single sphere
            num_new_spheres = 1
            spheres = spheres.broadcast_to((1,))
        else:
            num_new_spheres = attach_batch_axes[0]

        # Check if the current collision geometry is Sphere-based
        if not isinstance(self.coll, Sphere):
            raise TypeError("attach_link only works with Sphere-based collision models")

        # Get current collision data
        coll_batch_axes = self.coll.get_batch_axes()
        if len(coll_batch_axes) == 1:
            raise ValueError(
                "Cannot attach to simple collision model. Use from_urdf_spheres instead.")

        num_links, num_spheres_per_link = coll_batch_axes

        # Calculate new max spheres per link
        new_max_spheres = max(num_new_spheres, num_spheres_per_link)

        # Rebuild collision geometry with new link
        new_sphere_list = []
        new_num_spheres_per_link = list(self.num_spheres_per_link) if self.num_spheres_per_link else [
            num_spheres_per_link] * num_links

        # Add existing links with repadding if necessary
        for i in range(num_links):
            link_spheres = jax.tree.map(lambda x: x[i], self.coll)

            if self.num_spheres_per_link:
                valid_count = self.num_spheres_per_link[i]
                center = link_spheres.pose.translation()[:valid_count]
                radius = link_spheres.radius[:valid_count]
            else:
                center = link_spheres.pose.translation()
                radius = link_spheres.radius

            # Pad to new max
            if center.shape[0] < new_max_spheres:
                pad = new_max_spheres - center.shape[0]
                center = jnp.concatenate(
                    [center, jnp.zeros((pad, 3), dtype=center.dtype)], axis=0)
                radius = jnp.concatenate(
                    [radius, jnp.zeros((pad,), dtype=radius.dtype)], axis=0)

            new_sphere_list.append(Sphere.from_center_and_radius(center, radius))

        # Add new link
        center = spheres.pose.translation()
        radius = spheres.radius

        # Apply offset transformation if provided
        if offset is not None:
            # Transform sphere centers by the offset
            center_homogeneous = jnp.concatenate(
                [center, jnp.ones((center.shape[0], 1), dtype=center.dtype)], axis=-1)
            offset_matrix = offset.as_matrix()
            center_transformed = (offset_matrix @ center_homogeneous.T).T
            center = center_transformed[:, :3]

        if center.shape[0] < new_max_spheres:
            pad = new_max_spheres - center.shape[0]
            center = jnp.concatenate(
                [center, jnp.zeros((pad, 3), dtype=center.dtype)], axis=0)
            radius = jnp.concatenate(
                [radius, jnp.zeros((pad,), dtype=radius.dtype)], axis=0)

        new_sphere_list.append(Sphere.from_center_and_radius(center, radius))
        new_num_spheres_per_link.append(num_new_spheres)

        # Stack all spheres
        new_coll = cast(Sphere, jax.tree.map(lambda *args: jnp.stack(args), *new_sphere_list))

        # Update link names
        new_link_names = self.link_names + (new_link_name,)
        new_num_links = num_links + 1

        # Update parent link indices
        if self.parent_link_indices is None:
            # Initialize with -1 for all original links (no parent tracking)
            new_parent_indices = tuple([-1] * num_links + [parent_idx])
        else:
            new_parent_indices = self.parent_link_indices + (parent_idx,)

        # Update active collision pairs
        # Add all pairs with the new link (except parent if ignore_self_collision=True)
        new_idx_i_list = list(self.active_idx_i)
        new_idx_j_list = list(self.active_idx_j)

        new_link_idx = new_num_links - 1
        for i in range(num_links):
            if ignore_self_collision and i == parent_idx:
                continue  # Skip parent-child collision
            # Add pair (i, new_link_idx) where i < new_link_idx
            new_idx_i_list.append(i)
            new_idx_j_list.append(new_link_idx)

        new_active_idx_i = jnp.array(new_idx_i_list, dtype=jnp.int32)
        new_active_idx_j = jnp.array(new_idx_j_list, dtype=jnp.int32)

        return RobotCollision(
            num_links=new_num_links,
            link_names=new_link_names,
            active_idx_i=new_active_idx_i,
            active_idx_j=new_active_idx_j,
            coll=new_coll,
            num_spheres_per_link=tuple(new_num_spheres_per_link),
            parent_link_indices=new_parent_indices,
        )

    def update_link_spheres(
        self,
        link_name: str,
        new_spheres: Sphere,
        offset: jaxlie.SE3 | None = None,
    ) -> "RobotCollision":
        """
        Update the collision spheres of an existing link without changing the model structure.
        This avoids JIT recompilation by keeping the same number of spheres.

        Args:
            link_name: Name of the link to update.
            new_spheres: New sphere collision geometry. Must have the same batch size
                        as the current spheres for this link to avoid recompilation.
            offset: Optional SE3 transformation to apply to the new spheres.

        Returns:
            New RobotCollision instance with updated spheres for the specified link.

        Raises:
            ValueError: If link not found or sphere count mismatch.
        """
        if link_name not in self.link_names:
            raise ValueError(f"Link '{link_name}' not found in robot collision model")

        if not isinstance(self.coll, Sphere):
            raise TypeError("update_link_spheres only works with Sphere-based collision models")

        if self.num_spheres_per_link is None:
            raise ValueError("Cannot update link without sphere count tracking")

        link_idx = self.link_names.index(link_name)
        current_max_spheres = self.coll.get_batch_axes()[-1]

        # Get new sphere data
        new_batch_axes = new_spheres.get_batch_axes()
        if len(new_batch_axes) == 0:
            num_new_spheres = 1
            new_spheres = new_spheres.broadcast_to((1,))
        else:
            num_new_spheres = new_batch_axes[0]

        # Verify sphere count matches to avoid recompilation
        current_num_spheres = self.num_spheres_per_link[link_idx]
        if num_new_spheres != current_num_spheres:
            raise ValueError(
                f"Sphere count mismatch: link '{link_name}' currently has {current_num_spheres} "
                f"spheres, but new_spheres has {num_new_spheres}. Counts must match to avoid "
                f"JIT recompilation."
            )

        # Extract new sphere centers and radii
        center = new_spheres.pose.translation()
        radius = new_spheres.radius

        # Apply offset transformation if provided
        if offset is not None:
            center_homogeneous = jnp.concatenate(
                [center, jnp.ones((center.shape[0], 1), dtype=center.dtype)], axis=-1)
            offset_matrix = offset.as_matrix()
            center_transformed = (offset_matrix @ center_homogeneous.T).T
            center = center_transformed[:, :3]

        # Pad to max spheres
        if center.shape[0] < current_max_spheres:
            pad = current_max_spheres - center.shape[0]
            center = jnp.concatenate(
                [center, jnp.zeros((pad, 3), dtype=center.dtype)], axis=0)
            radius = jnp.concatenate(
                [radius, jnp.zeros((pad,), dtype=radius.dtype)], axis=0)

        # Reconstruct collision geometry with updated spheres
        new_sphere_list = []
        for i in range(self.num_links):
            if i == link_idx:
                # Use new spheres for this link
                new_sphere_list.append(Sphere.from_center_and_radius(center, radius))
            else:
                # Keep existing spheres
                link_spheres = jax.tree.map(lambda x: x[i], self.coll)
                existing_center = link_spheres.pose.translation()
                existing_radius = link_spheres.radius
                new_sphere_list.append(
                    Sphere.from_center_and_radius(existing_center, existing_radius))

        # Stack all spheres
        new_coll = cast(Sphere, jax.tree.map(lambda *args: jnp.stack(args), *new_sphere_list))

        return RobotCollision(
            num_links=self.num_links,
            link_names=self.link_names,
            active_idx_i=self.active_idx_i,
            active_idx_j=self.active_idx_j,
            coll=new_coll,
            num_spheres_per_link=self.num_spheres_per_link,
            parent_link_indices=self.parent_link_indices,
        )

    def detach_link(
        self,
        link_name: str,
    ) -> "RobotCollision":
        """
        Detach (remove) a link from the collision model.

        Args:
            link_name: Name of the link to remove.

        Returns:
            New RobotCollision instance without the detached link.
        """
        if link_name not in self.link_names:
            raise ValueError(f"Link '{link_name}' not found in robot collision model")

        link_idx = self.link_names.index(link_name)

        # Check if the current collision geometry is Sphere-based
        if not isinstance(self.coll, Sphere):
            raise TypeError("detach_link only works with Sphere-based collision models")

        if self.num_spheres_per_link is None:
            raise ValueError("Cannot detach from model without sphere count tracking")

        # Get current collision data
        coll_batch_axes = self.coll.get_batch_axes()
        if len(coll_batch_axes) == 1:
            raise ValueError(
                "Cannot detach from simple collision model. Use from_urdf_spheres instead.")

        num_links, num_spheres_per_link = coll_batch_axes

        # Rebuild collision geometry without the target link
        new_sphere_list = []
        new_num_spheres_per_link = []
        new_link_names_list = []
        new_parent_indices_list = []

        # Create index mapping: old_idx -> new_idx
        idx_mapping = {}
        new_idx = 0
        for i in range(num_links):
            if i == link_idx:
                continue  # Skip the link to remove

            idx_mapping[i] = new_idx
            new_idx += 1

            # Extract and keep this link's spheres
            link_spheres = jax.tree.map(lambda x: x[i], self.coll)
            valid_count = self.num_spheres_per_link[i]
            center = link_spheres.pose.translation()[:valid_count]
            radius = link_spheres.radius[:valid_count]

            # Pad to current max
            if center.shape[0] < num_spheres_per_link:
                pad = num_spheres_per_link - center.shape[0]
                center = jnp.concatenate(
                    [center, jnp.zeros((pad, 3), dtype=center.dtype)], axis=0)
                radius = jnp.concatenate(
                    [radius, jnp.zeros((pad,), dtype=radius.dtype)], axis=0)

            new_sphere_list.append(Sphere.from_center_and_radius(center, radius))
            new_num_spheres_per_link.append(self.num_spheres_per_link[i])
            new_link_names_list.append(self.link_names[i])

            if self.parent_link_indices:
                old_parent = self.parent_link_indices[i]
                if old_parent == -1 or old_parent == link_idx:
                    new_parent_indices_list.append(-1)
                else:
                    new_parent_indices_list.append(idx_mapping.get(old_parent, -1))

        if not new_sphere_list:
            raise ValueError("Cannot remove all links from collision model")

        # Stack all remaining spheres
        new_coll = cast(Sphere, jax.tree.map(lambda *args: jnp.stack(args), *new_sphere_list))

        # Update active collision pairs
        new_idx_i_list = []
        new_idx_j_list = []

        for i, j in zip(self.active_idx_i, self.active_idx_j):
            if i == link_idx or j == link_idx:
                continue  # Skip pairs involving the removed link

            new_i = idx_mapping[int(i)]
            new_j = idx_mapping[int(j)]
            new_idx_i_list.append(new_i)
            new_idx_j_list.append(new_j)

        new_active_idx_i = jnp.array(new_idx_i_list, dtype=jnp.int32)
        new_active_idx_j = jnp.array(new_idx_j_list, dtype=jnp.int32)

        return RobotCollision(
            num_links=len(new_link_names_list),
            link_names=tuple(new_link_names_list),
            active_idx_i=new_active_idx_i,
            active_idx_j=new_active_idx_j,
            coll=new_coll,
            num_spheres_per_link=tuple(new_num_spheres_per_link),
            parent_link_indices=tuple(
                new_parent_indices_list) if self.parent_link_indices else None,
        )

    @staticmethod
    def from_urdf(
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...] = (),
        ignore_immediate_adjacents: bool = True,
    ):
        """
        Build a differentiable robot collision model from a URDF.

        Args:
            urdf: The URDF object (used to load collision meshes).
            user_ignore_pairs: Additional pairs of link names to ignore for self-collision.
            ignore_immediate_adjacents: If True, automatically ignore collisions
                between adjacent (parent/child) links based on the URDF structure.
        """
        # Re-load urdf with collision data if not already loaded.
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access
        try:
            has_collision = any(link.collisions for link in urdf.link_map.values())
            if not has_collision:
                urdf = yourdfpy.URDF(
                    robot=urdf.robot,
                    filename_handler=filename_handler,
                    load_collision_meshes=True,
                )
        except Exception as e:
            logger.warning(f"Could not reload URDF with collision meshes: {e}")

        _, link_info = RobotURDFParser.parse(urdf)
        link_name_list = link_info.names  # Use names from parser

        # Gather all collision meshes.
        # The order of cap_list must match link_name_list.
        cap_list = list[Capsule]()
        for link_name in link_name_list:
            cap_list.append(
                Capsule.from_trimesh(
                    RobotCollision._get_trimesh_collision_geometries(urdf, link_name)
                )
            )

        # Convert list of trimesh objects into a batched Capsule object.
        capsules = cast(Capsule, jax.tree.map(lambda *args: jnp.stack(args), *cap_list))
        assert capsules.get_batch_axes() == (link_info.num_links,)

        # Directly compute active pair indices
        active_idx_i, active_idx_j = RobotCollision._compute_active_pair_indices(
            link_names=link_name_list,
            urdf=urdf,
            user_ignore_pairs=user_ignore_pairs,
            ignore_immediate_adjacents=ignore_immediate_adjacents,
        )

        logger.info(
            f"Created RobotCollision with {link_info.num_links} links and "
            f"{len(active_idx_i)} active self-collision pairs."
        )

        return RobotCollision(
            num_links=link_info.num_links,
            link_names=link_name_list,
            active_idx_i=active_idx_i,
            active_idx_j=active_idx_j,
            coll=capsules,
        )

    @staticmethod
    def _compute_active_pair_indices(
        link_names: tuple[str, ...],
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...],
        ignore_immediate_adjacents: bool,
    ) -> Tuple[Int[Array, " P"], Int[Array, " P"]]:
        """
        Computes the indices (i, j) of pairs where i < j and the pair should
        be actively checked for self-collision.

        Args:
            link_names: Tuple of link names in order.
            urdf: Parsed URDF object.
            user_ignore_pairs: List of (name1, name2) pairs to explicitly ignore.
            ignore_immediate_adjacents: Whether to ignore parent-child pairs from URDF.

        Returns:
            Tuple of (active_i, active_j) index arrays.
        """
        # --- Start: Logic combined from _build_ignore_matrix --- #
        num_links = len(link_names)
        link_name_to_idx = {name: i for i, name in enumerate(link_names)}
        ignore_matrix = jnp.zeros((num_links, num_links), dtype=bool)
        ignore_matrix = ignore_matrix.at[
            jnp.arange(num_links), jnp.arange(num_links)
        ].set(True)
        if ignore_immediate_adjacents:
            for joint in urdf.joint_map.values():
                parent_name = joint.parent
                child_name = joint.child
                if parent_name in link_name_to_idx and child_name in link_name_to_idx:
                    parent_idx = link_name_to_idx[parent_name]
                    child_idx = link_name_to_idx[child_name]
                    ignore_matrix = ignore_matrix.at[parent_idx, child_idx].set(True)
                    ignore_matrix = ignore_matrix.at[child_idx, parent_idx].set(True)
        for name1, name2 in user_ignore_pairs:
            if name1 in link_name_to_idx and name2 in link_name_to_idx:
                idx1 = link_name_to_idx[name1]
                idx2 = link_name_to_idx[name2]
                ignore_matrix = ignore_matrix.at[idx1, idx2].set(True)
                ignore_matrix = ignore_matrix.at[idx2, idx1].set(True)

        idx_i, idx_j = jnp.tril_indices(num_links, k=-1)
        should_check = ~ignore_matrix[idx_i, idx_j]
        active_i = idx_i[should_check]
        active_j = idx_j[should_check]

        return active_i, active_j

    @staticmethod
    def _get_trimesh_collision_geometries(
        urdf: yourdfpy.URDF, link_name: str
    ) -> trimesh.Trimesh:
        """Extracts trimesh collision geometries for a given link name, applying relative transforms."""
        if link_name not in urdf.link_map:
            return trimesh.Trimesh()

        link = urdf.link_map[link_name]
        filename_handler = urdf._filename_handler
        coll_meshes = []

        for collision in link.collisions:
            geom = collision.geometry
            mesh: Optional[trimesh.Trimesh] = None

            # Get the transform of the collision geometry relative to the link frame
            if collision.origin is not None:
                transform = collision.origin
            else:
                transform = jaxlie.SE3.identity().as_matrix()

            if geom.box is not None:
                mesh = trimesh.creation.box(extents=geom.box.size)
            elif geom.cylinder is not None:
                mesh = trimesh.creation.cylinder(
                    radius=geom.cylinder.radius, height=geom.cylinder.length
                )
            elif geom.sphere is not None:
                # print(f'{link_name}, {geom.sphere.radius}')
                mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
            elif geom.mesh is not None:
                try:
                    mesh_path = geom.mesh.filename
                    loaded_obj = trimesh.load(
                        file_obj=filename_handler(mesh_path), force="mesh"
                    )

                    scale = (
                        geom.mesh.scale
                        if geom.mesh.scale is not None
                        else [1.0, 1.0, 1.0]
                    )

                    if isinstance(loaded_obj, trimesh.Trimesh):
                        mesh = loaded_obj.copy()
                        mesh.apply_scale(scale)
                    elif isinstance(loaded_obj, trimesh.Scene):
                        if len(loaded_obj.geometry) > 0:
                            geom_candidate = list(loaded_obj.geometry.values())[0]
                            if isinstance(geom_candidate, trimesh.Trimesh):
                                mesh = geom_candidate.copy()
                                mesh.apply_scale(scale)
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue  # Skip if load result is unexpected

                    if mesh:
                        mesh.fix_normals()

                except Exception as e:
                    logger.error(
                        f"Failed processing mesh '{geom.mesh.filename}' for link '{link_name}': {e}"
                    )
                    continue
            else:
                logger.warning(
                    f"Unsupported collision geometry type for link '{link_name}'."
                )
                continue

            if mesh is not None:
                # Apply the transform specified in the URDF collision tag
                mesh.apply_transform(transform)
                coll_meshes.append(mesh)

        coll_mesh = sum(coll_meshes, trimesh.Trimesh())
        return coll_mesh

    @staticmethod
    def from_urdf_spheres(
        urdf: yourdfpy.URDF,
        user_ignore_pairs: tuple[tuple[str, str], ...] = (),
        ignore_immediate_adjacents: bool = True,
    ):
        """
        Build a differentiable robot collision model from a URDF.

        Args:
            urdf: The URDF object (used to load collision meshes).
            user_ignore_pairs: Additional pairs of link names to ignore for self-collision.
            ignore_immediate_adjacents: If True, automatically ignore collisions
                between adjacent (parent/child) links based on the URDF structure.
        """
        # Re-load urdf with collision data if not already loaded.
        filename_handler = urdf._filename_handler  # pylint: disable=protected-access
        try:
            has_collision = any(link.collisions for link in urdf.link_map.values())
            if not has_collision:
                urdf = yourdfpy.URDF(
                    robot=urdf.robot,
                    filename_handler=filename_handler,
                    load_collision_meshes=True,
                )
        except Exception as e:
            logger.warning(f"Could not reload URDF with collision meshes: {e}")

        _, link_info = RobotURDFParser.parse(urdf)
        link_name_list = link_info.names  # Use names from parser

        # # Gather all collision meshes.
        # # The order of cap_list must match link_name_list.
        # sphere_list = list[Sphere]()
        # for link_name in link_name_list:
        #     sphere_list.append(_get_sphere_collision_geometries(urdf, link_name))

        # # Convert list of trimesh objects into a batched Capsule object.
        # spheres = cast(Sphere, jax.tree.map(lambda *args: jnp.stack(args), *sphere_list))

        # sphere_list = []
        # for link_name in link_name_list:
        #     s = RobotCollision._get_sphere_collision_geometries(urdf, link_name)
        #     if s is None:
        #         s = Sphere.from_center_and_radius(jnp.zeros(3), 0.0)
        #     sphere_list.append(s)

        # spheres = cast(Sphere, jax.tree.map(lambda *args: jnp.stack(args),
        #                                     *sphere_list))
        # まず各リンクの球数を調べる
        num_spheres_per_link = []
        for link_name in link_name_list:
            s = RobotCollision._get_sphere_collision_geometries(urdf, link_name)
            if s is None:
                num_spheres_per_link.append(0)
            else:
                num_spheres_per_link.append(s.get_batch_axes()[0])
        max_spheres = max(num_spheres_per_link)

        # 各リンクの球をmax_spheres個に揃えてリスト化
        sphere_list = []
        for link_name in link_name_list:
            s = RobotCollision._get_sphere_collision_geometries(urdf, link_name)
            if s is None or s.get_batch_axes()[0] == 0:
                # ダミー
                center = jnp.zeros((max_spheres, 3), dtype=jnp.float32)
                radius = jnp.zeros((max_spheres,), dtype=jnp.float32)
                s = Sphere.from_center_and_radius(center, radius)
            else:
                # パディング
                n = s.get_batch_axes()[0]
                if n < max_spheres:
                    pad = max_spheres - n
                    center = jnp.concatenate([s.pose.translation(), jnp.zeros(
                        (pad, 3), dtype=s.pose.translation().dtype)], axis=0)
                    radius = jnp.concatenate(
                        [s.radius, jnp.zeros((pad,), dtype=s.radius.dtype)], axis=0)
                    s = Sphere.from_center_and_radius(center, radius)
            sphere_list.append(s)

        spheres = cast(Sphere, jax.tree.map(lambda *args: jnp.stack(args), *sphere_list))

        # Directly compute active pair indices
        active_idx_i, active_idx_j = RobotCollision._compute_active_pair_indices(
            link_names=link_name_list,
            urdf=urdf,
            user_ignore_pairs=user_ignore_pairs,
            ignore_immediate_adjacents=ignore_immediate_adjacents,
        )

        logger.info(
            f"Created RobotCollision with {link_info.num_links} links and "
            f"{len(active_idx_i)} active self-collision pairs."
        )

        return RobotCollision(
            num_links=link_info.num_links,
            link_names=link_name_list,
            active_idx_i=active_idx_i,
            active_idx_j=active_idx_j,
            coll=spheres,
            num_spheres_per_link=tuple(num_spheres_per_link),
        )

    @staticmethod
    def _get_sphere_collision_geometries(
        urdf: yourdfpy.URDF, link_name: str
    ) -> Sphere | None:
        """Extracts sphere collision geometries for a given link name, applying relative transforms."""
        link = urdf.link_map[link_name]
        radius = []
        pts = []

        for collision in link.collisions:
            geom = collision.geometry

            # Get the transform of the collision geometry relative to the link frame
            if collision.origin is not None:
                transform = collision.origin
                pts.append(transform[:3, 3])
            else:
                transform = jaxlie.SE3.identity().as_matrix()
                pts.append(transform[:3, 3])

            if geom.sphere is not None:
                radius.append(geom.sphere.radius)
            else:
                logger.warning(
                    f"No sphere geometry type for link '{link_name}'."
                )
                continue

        if radius:
            spheres = Sphere.from_center_and_radius(center=jnp.array(pts), radius=jnp.array(radius))
            return spheres
        else:
            return None

    @jdc.jit
    def at_config(
        self, robot: Robot, cfg: Float[Array, "*batch actuated_count"]
    ) -> CollGeom:
        """
        Returns the collision geometry transformed to the given robot configuration.

        Ensures that the link transforms returned by forward kinematics are applied
        to the corresponding collision geometries stored in this object, based on link names.

        For attached links (with parent_link_indices), uses the parent link's transformation.

        Args:
            robot: The Robot instance containing kinematics information.
            cfg: The robot configuration (actuated joints).

        Returns:
            The collision geometry (CollGeom) transformed to the world frame
            according to the provided configuration.
        """
        # Get transforms for robot's kinematic links
        Ts_link_world_wxyz_xyz = robot.forward_kinematics(cfg)
        Ts_robot_links = jaxlie.SE3(Ts_link_world_wxyz_xyz)

        # Build transform list for all collision links (including attached ones)
        if self.parent_link_indices is not None and len(self.link_names) > len(robot.links.names):
            # We have attached links - need to map them to their parent transforms
            Ts_all_links_list = []

            for i, link_name in enumerate(self.link_names):
                if link_name in robot.links.names:
                    # Original robot link - use FK result
                    robot_link_idx = robot.links.names.index(link_name)
                    # Extract individual SE3 from batched SE3
                    T_link = jax.tree.map(lambda x: x[robot_link_idx], Ts_robot_links)
                    Ts_all_links_list.append(T_link)
                else:
                    # Attached link - use parent link's transform
                    parent_idx = self.parent_link_indices[i]
                    if parent_idx >= 0:
                        Ts_all_links_list.append(Ts_all_links_list[parent_idx])
                    else:
                        # No parent - use identity (shouldn't happen for attached links)
                        Ts_all_links_list.append(jaxlie.SE3.identity())

            # Stack into array
            Ts_link_world = jax.tree.map(lambda *args: jnp.stack(args), *Ts_all_links_list)
        else:
            # No attached links - original behavior
            assert self.link_names == robot.links.names, (
                "Link name mismatch between RobotCollision and Robot kinematics."
            )
            # Handle nested batch dimensions (e.g., multiple spheres per link)
            Ts_link_world = Ts_robot_links
        coll_batch_axes = self.coll.get_batch_axes()
        if len(coll_batch_axes) > 1:
            # Expand transform dimensions to match collision geometry
            # Transform shape: (num_links,) -> (num_links, 1, ..., 1)
            expand_dims = tuple([slice(None)] + [None] * (len(coll_batch_axes) - 1))
            Ts_expanded = jax.tree.map(lambda x: x[expand_dims], Ts_link_world)
            return self.coll.transform(Ts_expanded)
        else:
            return self.coll.transform(Ts_link_world)

    def get_swept_capsules(
        self,
        robot: Robot,
        cfg_prev: Float[Array, "*batch actuated_count"],
        cfg_next: Float[Array, "*batch actuated_count"],
    ) -> Capsule:
        """
        Computes swept-volume capsules between two configurations.

        For each link, the capsule at cfg_prev and cfg_next is decomposed into
        a fixed number of spheres (currently 5). Corresponding sphere pairs are
        then connected by capsules to represent the swept volume.

        Args:
            robot: The Robot instance.
            cfg_prev: The starting robot configuration.
            cfg_next: The ending robot configuration.

        Returns:
            A Capsule object representing the swept volumes.
            The batch axes will be (*batch, 5, num_links).
        """
        n_segments = 5

        # 1. Get collision geometries at start and end configurations
        # Shape: (*batch, num_links)
        coll_prev_world: Capsule = cast(Capsule, self.at_config(robot, cfg_prev))
        coll_next_world: Capsule = cast(Capsule, self.at_config(robot, cfg_next))
        assert isinstance(coll_prev_world, Capsule)
        assert isinstance(coll_next_world, Capsule)
        assert coll_prev_world.get_batch_axes() == coll_next_world.get_batch_axes()

        # 2. Decompose capsules into spheres
        # Shape: (n_segments, *batch, num_links)
        spheres_prev = coll_prev_world.decompose_to_spheres(n_segments)
        spheres_next = coll_next_world.decompose_to_spheres(n_segments)
        assert spheres_prev.get_batch_axes() == spheres_next.get_batch_axes(), (
            "Sphere batch axes mismatch after decomposition."
        )
        expected_sphere_batch_axes = (
            (n_segments,) + cfg_prev.shape[:-1] + (self.num_links,)
        )
        assert spheres_prev.get_batch_axes() == expected_sphere_batch_axes, (
            f"Unexpected sphere batch axes: {spheres_prev.get_batch_axes()} vs {expected_sphere_batch_axes}"
        )

        # 3. Create swept capsules by connecting corresponding sphere pairs
        # Shape: (n_segments, *batch, num_links)
        swept_capsules = Capsule.from_sphere_pairs(spheres_prev, spheres_next)
        assert swept_capsules.get_batch_axes() == expected_sphere_batch_axes, (
            "Swept capsule batch axes mismatch."
        )

        # The result contains capsules for each segment of each link.
        return swept_capsules

    def compute_self_collision_distance(
        self,
        robot: Robot,
        cfg: Float[Array, "*batch actuated_count"],
    ) -> Float[Array, "*batch num_active_pairs"]:
        """
        Computes the signed distances for active self-collision pairs.

        Args:
            robot_coll: The robot's collision model with precomputed active pair indices.
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).

        Returns:
            Signed distances for each active pair.
            Shape: (*batch, num_active_pairs).
            Positive distance means separation, negative means penetration.
        """
        batch_axes = cfg.shape[:-1]

        # 1. Get collision geometry at the current config
        coll = self.at_config(robot, cfg)
        coll_batch_axes = coll.get_batch_axes()

        # Handle nested batch dimensions (e.g., multiple spheres per link)
        if len(coll_batch_axes) > len(batch_axes) + 1:
            # Flatten nested dimensions: (batch, num_links, num_spheres) -> (batch, num_links*num_spheres)
            nested_shape = coll_batch_axes[len(batch_axes):]
            flattened_size = int(jnp.prod(jnp.array(nested_shape)))
            coll = coll.reshape((*batch_axes, flattened_size))

        assert coll.get_batch_axes() == (
            *batch_axes, self.num_links) or len(coll.get_batch_axes()) == len(batch_axes) + 1

        # 2. Compute all pairwise distances using the imported function
        dist_matrix = pairwise_collide(coll, coll)

        # For flattened case, we need to compute minimum distances between link groups
        if len(coll_batch_axes) > len(batch_axes) + 1:
            # Reshape distance matrix to group by links
            num_spheres_per_link_max = coll_batch_axes[-1]
            dist_matrix_reshaped = dist_matrix.reshape(
                *batch_axes, self.num_links, num_spheres_per_link_max,
                self.num_links, num_spheres_per_link_max
            )

            # If num_spheres_per_link info is available, mask out padding spheres
            if self.num_spheres_per_link is not None:
                # Create mask for valid spheres (non-padding)
                mask_i = jnp.array([
                    [i < self.num_spheres_per_link[link]
                     for i in range(num_spheres_per_link_max)]
                    for link in range(self.num_links)
                ])  # Shape: (num_links, num_spheres_per_link_max)

                mask_j = mask_i  # Same mask for both dimensions

                # Broadcast masks to match dist_matrix_reshaped shape
                # Shape: (num_links, num_spheres_max, num_links, num_spheres_max)
                mask_combined = mask_i[:, :, None, None] & mask_j[None, None, :, :]

                # Set invalid (padding) distances to a large value (inf)
                dist_matrix_reshaped = jnp.where(
                    mask_combined,
                    dist_matrix_reshaped,
                    jnp.inf
                )

            # Take minimum distance between any pair of valid spheres from two different links
            dist_matrix = jnp.min(dist_matrix_reshaped, axis=(-1, -3))

        assert dist_matrix.shape == (
            *batch_axes,
            self.num_links,
            self.num_links,
        )

        # 3. Extract distances for the precomputed active pairs
        # Use advanced indexing with the stored indices
        active_distances = dist_matrix[..., self.active_idx_i, self.active_idx_j]

        # Expected shape check
        num_active_pairs = len(self.active_idx_i)
        assert active_distances.shape == (*batch_axes, num_active_pairs)

        return active_distances

    def compute_world_collision_distance(
        self,
        robot: Robot,
        cfg: Float[Array, "*batch_cfg actuated_count"],
        world_geom: CollGeom,  # Shape: (*batch_world, M, ...)
    ) -> Float[Array, "*batch_combined N M"]:
        """
        Computes the signed distances between all robot links (N) and all world obstacles (M).

        Args:
            robot_coll: The robot's collision model.
            robot: The robot's kinematic model.
            cfg: The robot configuration (actuated joints).
            world_geom: Collision geometry representing world obstacles. If representing a
                single obstacle, it should have batch shape (). If multiple, the last axis
                is interpreted as the collection of world objects (M).
                The batch dimensions (*batch_world) must be broadcast-compatible with cfg's
                batch axes (*batch_cfg).

        Returns:
            Matrix of signed distances between each robot link and each world object.
            Shape: (*batch_combined, N, M), where N=num_links, M=num_world_objects.
            Positive distance means separation, negative means penetration.
        """
        batch_cfg_shape = cfg.shape[:-1]

        # 1. Get robot collision geometry at the current config
        # Shape: (*batch_cfg, N, ...) or (*batch_cfg, N, num_spheres)
        coll_robot_world = self.at_config(robot, cfg)
        N = self.num_links
        coll_batch_axes = coll_robot_world.get_batch_axes()

        # Determine if we have nested dimensions based on static information
        # Handle nested batch dimensions (e.g., multiple spheres per link)
        has_nested_dims = len(coll_batch_axes) > len(batch_cfg_shape) + 1
        if has_nested_dims:
            # Flatten nested dimensions: (*batch, num_links, num_spheres) -> (*batch, num_links*num_spheres)
            nested_shape = coll_batch_axes[len(batch_cfg_shape):]
            # Use static shape calculation
            flattened_size = int(onp.prod(nested_shape))
            num_spheres_per_link_max = nested_shape[-1]
            coll_robot_world = coll_robot_world.reshape((*batch_cfg_shape, flattened_size))
            coll_batch_axes = coll_robot_world.get_batch_axes()
        else:
            num_spheres_per_link_max = None

        assert coll_robot_world.get_batch_axes()[-1] == N or has_nested_dims
        batch_cfg_shape = coll_robot_world.get_batch_axes()[:-1]

        # 2. Normalize world_geom shape and determine M
        world_axes = world_geom.get_batch_axes()
        if len(world_axes) == 0:  # Single world object
            # Use the object's broadcast_to method to add the M=1 axis correctly
            _world_geom = world_geom.broadcast_to((1,))
            M = 1
            batch_world_shape = ()
        else:  # Multiple world objects
            _world_geom = world_geom
            M = world_axes[-1]
            batch_world_shape = world_axes[:-1]

        # 3. Compute distances
        if has_nested_dims:
            # For flattened spheres: compute all distances then group by links
            _collide_links_vs_world = jax.vmap(collide, in_axes=(-2, None), out_axes=(-2))
            dist_matrix_flat = _collide_links_vs_world(coll_robot_world, _world_geom)

            # Reshape to group by links and take minimum per link
            dist_matrix_reshaped = dist_matrix_flat.reshape(
                *batch_cfg_shape, N, num_spheres_per_link_max, M
            )
            if self.num_spheres_per_link is not None:
                mask = jnp.array([
                    [i < self.num_spheres_per_link[link]
                     for i in range(num_spheres_per_link_max)]
                    for link in range(N)
                ])  # Shape: (N, num_spheres_max)

                # Broadcast mask: (N, num_spheres_max) -> (N, num_spheres_max, 1)
                mask_expanded = mask[:, :, None]

                # Set invalid distances to inf
                dist_matrix_reshaped = jnp.where(
                    mask_expanded,
                    dist_matrix_reshaped,
                    jnp.inf
                )

            # Take minimum distance per link
            dist_matrix = jnp.min(dist_matrix_reshaped, axis=-2)
        else:
            # Original logic for single collision per link
            _collide_links_vs_world = jax.vmap(collide, in_axes=(-2, None), out_axes=(-2))
            dist_matrix = _collide_links_vs_world(coll_robot_world, _world_geom)

        # 4. Result shape check
        # Calculate expected shape based on broadcasting rules
        expected_batch_combined = jnp.broadcast_shapes(
            batch_cfg_shape, batch_world_shape
        )
        expected_shape = (*expected_batch_combined, N, M)

        # Perform the assertion without try-except or complex logic
        assert dist_matrix.shape == expected_shape, (
            f"Output shape mismatch. Expected {expected_shape}, Got {dist_matrix.shape}. "
            f"Robot axes: {coll_robot_world.get_batch_axes()}, Original World axes: {world_geom.get_batch_axes()}"
        )

        # 5. Return the distance matrix
        return dist_matrix
