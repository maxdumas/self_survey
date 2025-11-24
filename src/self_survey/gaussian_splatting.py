"""
3D Gaussian Splatting export and training from LiDAR point clouds.

This module provides functionality to:
1. Run COLMAP for camera pose estimation from photos
2. Register COLMAP's reconstruction to LiDAR coordinates via ICP
3. Initialize 3D Gaussians from LiDAR points
4. Train the Gaussian Splatting model
5. Export to standard PLY format

Pipeline Overview
-----------------

The pipeline correlates photos to LiDAR through geometric registration:

1. **COLMAP** processes photos to estimate camera poses and produces a sparse
   3D reconstruction (Structure from Motion).

2. **ICP Registration** aligns COLMAP's sparse point cloud to the LiDAR point
   cloud. Both capture the same scene geometry, so ICP can find the rigid
   transformation between coordinate systems.

3. **Camera Pose Transformation** applies the same ICP transformation to all
   camera poses, bringing them into the LiDAR coordinate system.

4. **Gaussian Initialization** creates initial 3D Gaussians at each LiDAR
   point location, with colors from the point cloud (if available).

5. **Training** optimizes Gaussian parameters (position, covariance, color,
   opacity) to match the input photos using differentiable rendering.

Why LiDAR Initialization Helps
------------------------------

Standard 3DGS initializes from COLMAP's sparse points (typically thousands).
LiDAR provides millions of accurately-positioned points, which:

- Reduces optimization time (geometry is already good)
- Eliminates "floater" artifacts (spurious Gaussians in empty space)
- Provides better coverage in texture-poor regions
- Preserves accurate geometry from the LiDAR survey

Device Support
--------------

Training supports CUDA (fastest), MPS (Apple Silicon), and CPU (slow).
The gsplat library handles device-specific rasterization.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Camera:
    """Camera intrinsics and extrinsics."""

    id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # Extrinsics: world-to-camera transformation
    qvec: NDArray[np.float64]  # quaternion (w, x, y, z)
    tvec: NDArray[np.float64]  # translation
    image_path: Path | None = None

    @property
    def intrinsic_matrix(self) -> NDArray[np.float64]:
        """3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ])

    @property
    def rotation_matrix(self) -> NDArray[np.float64]:
        """3x3 rotation matrix from quaternion."""
        return qvec_to_rotmat(self.qvec)

    @property
    def camera_center(self) -> NDArray[np.float64]:
        """Camera center in world coordinates."""
        R = self.rotation_matrix
        return -R.T @ self.tvec


@dataclass
class ColmapReconstruction:
    """COLMAP sparse reconstruction data."""

    cameras: list[Camera]
    points3d: NDArray[np.float64]  # (N, 3) world coordinates
    point_colors: NDArray[np.float64] | None = None  # (N, 3) RGB [0, 1]
    workspace_path: Path | None = None


@dataclass
class GaussianModel:
    """3D Gaussian Splatting model parameters."""

    positions: NDArray[np.float64]  # (N, 3) centers
    colors: NDArray[np.float64]  # (N, 3) RGB or SH coefficients
    opacities: NDArray[np.float64]  # (N,) in logit space
    scales: NDArray[np.float64]  # (N, 3) log-scale
    rotations: NDArray[np.float64]  # (N, 4) quaternions

    @classmethod
    def from_points(
        cls,
        points: NDArray[np.float64],
        colors: NDArray[np.float64] | None = None,
        initial_scale: float = 0.1,
    ) -> GaussianModel:
        """
        Initialize Gaussians from point cloud.

        Parameters
        ----------
        points : ndarray of shape (N, 3)
            Point positions
        colors : ndarray of shape (N, 3), optional
            RGB colors in [0, 1]. If None, defaults to gray.
        initial_scale : float
            Initial Gaussian scale (log-space)
        """
        n_points = len(points)

        if colors is None:
            colors = np.full((n_points, 3), 0.5)

        # Initialize scales based on local point density
        scales = np.full((n_points, 3), np.log(initial_scale))

        # Identity rotations (w=1, x=y=z=0)
        rotations = np.zeros((n_points, 4))
        rotations[:, 0] = 1.0

        # Initialize opacity (sigmoid of 0 = 0.5)
        opacities = np.zeros(n_points)

        return cls(
            positions=points.astype(np.float64),
            colors=colors.astype(np.float64),
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )


def check_colmap_installed() -> bool:
    """Check if COLMAP is installed and accessible."""
    return shutil.which("colmap") is not None


def run_colmap(
    image_dir: Path,
    workspace_dir: Path,
    camera_model: str = "SIMPLE_RADIAL",
    use_gpu: bool = True,
) -> ColmapReconstruction:
    """
    Run COLMAP Structure-from-Motion pipeline on images.

    Parameters
    ----------
    image_dir : Path
        Directory containing input images
    workspace_dir : Path
        Output directory for COLMAP database and reconstruction
    camera_model : str
        Camera model to use (SIMPLE_RADIAL, PINHOLE, OPENCV, etc.)
    use_gpu : bool
        Use GPU for feature extraction/matching if available

    Returns
    -------
    ColmapReconstruction
        Cameras, poses, and sparse 3D points
    """
    if not check_colmap_installed():
        raise RuntimeError(
            "COLMAP is not installed. Please install it:\n"
            "  macOS: brew install colmap\n"
            "  Ubuntu: sudo apt install colmap\n"
            "  Windows: Download from https://colmap.github.io/"
        )

    workspace_dir.mkdir(parents=True, exist_ok=True)
    database_path = workspace_dir / "database.db"
    sparse_dir = workspace_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    gpu_flag = "1" if use_gpu else "0"

    # Step 1: Feature extraction
    print("\n  Running feature extraction...")
    subprocess.run(
        [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", gpu_flag,
        ],
        check=True,
        capture_output=True,
    )

    # Step 2: Feature matching
    print("  Running feature matching...")
    subprocess.run(
        [
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
            "--SiftMatching.use_gpu", gpu_flag,
        ],
        check=True,
        capture_output=True,
    )

    # Step 3: Sparse reconstruction
    print("  Running sparse reconstruction...")
    subprocess.run(
        [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
        ],
        check=True,
        capture_output=True,
    )

    # Find the reconstruction (usually in sparse/0)
    recon_dirs = list(sparse_dir.iterdir())
    if not recon_dirs:
        raise RuntimeError("COLMAP reconstruction failed - no output produced")

    recon_dir = recon_dirs[0]  # Use first reconstruction

    # Convert to text format for easier parsing
    text_dir = workspace_dir / "sparse_text"
    text_dir.mkdir(exist_ok=True)

    subprocess.run(
        [
            "colmap", "model_converter",
            "--input_path", str(recon_dir),
            "--output_path", str(text_dir),
            "--output_type", "TXT",
        ],
        check=True,
        capture_output=True,
    )

    # Parse the reconstruction
    reconstruction = parse_colmap_text(text_dir, image_dir)
    reconstruction.workspace_path = workspace_dir

    return reconstruction


def parse_colmap_text(
    text_dir: Path,
    image_dir: Path,
) -> ColmapReconstruction:
    """Parse COLMAP text format reconstruction."""
    cameras_file = text_dir / "cameras.txt"
    images_file = text_dir / "images.txt"
    points_file = text_dir / "points3D.txt"

    # Parse cameras.txt (intrinsics)
    camera_intrinsics = {}
    with open(cameras_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]

            # Handle different camera models
            if model == "SIMPLE_RADIAL":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            elif model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            else:
                # Default to first params
                fx = fy = params[0]
                cx, cy = width / 2, height / 2

            camera_intrinsics[cam_id] = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }

    # Parse images.txt (extrinsics)
    cameras = []
    with open(images_file) as f:
        lines = [l for l in f if not l.startswith("#")]

    # images.txt has pairs of lines: image info, then 2D points
    for i in range(0, len(lines), 2):
        if i >= len(lines):
            break
        parts = lines[i].strip().split()
        if len(parts) < 10:
            continue

        image_id = int(parts[0])
        qvec = np.array([float(parts[j]) for j in range(1, 5)])
        tvec = np.array([float(parts[j]) for j in range(5, 8)])
        cam_id = int(parts[8])
        image_name = parts[9]

        intrinsics = camera_intrinsics.get(cam_id, camera_intrinsics[1])

        cameras.append(Camera(
            id=image_id,
            width=intrinsics["width"],
            height=intrinsics["height"],
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            qvec=qvec,
            tvec=tvec,
            image_path=image_dir / image_name,
        ))

    # Parse points3D.txt
    points = []
    colors = []
    with open(points_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])

    return ColmapReconstruction(
        cameras=cameras,
        points3d=np.array(points) if points else np.zeros((0, 3)),
        point_colors=np.array(colors) if colors else None,
    )


def register_colmap_to_lidar(
    colmap_points: NDArray[np.float64],
    lidar_points: NDArray[np.float64],
    max_correspondence_distance: float = 10.0,
    max_iterations: int = 50,
) -> NDArray[np.float64]:
    """
    Register COLMAP reconstruction to LiDAR coordinates using ICP.

    Parameters
    ----------
    colmap_points : ndarray of shape (N, 3)
        Sparse points from COLMAP
    lidar_points : ndarray of shape (M, 3)
        Dense points from LiDAR
    max_correspondence_distance : float
        Maximum distance for point correspondences
    max_iterations : int
        Maximum ICP iterations

    Returns
    -------
    transform : ndarray of shape (4, 4)
        Transformation matrix to apply to COLMAP coordinates
    """
    import open3d as o3d

    # Create point clouds
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(colmap_points)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(lidar_points)

    # Estimate normals for point-to-plane ICP
    target.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )

    # Initial alignment using centroids
    source_center = np.mean(colmap_points, axis=0)
    target_center = np.mean(lidar_points, axis=0)
    initial_transform = np.eye(4)
    initial_transform[:3, 3] = target_center - source_center

    # Run ICP
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )

    print(f"  ICP fitness: {result.fitness:.4f}")
    print(f"  ICP RMSE: {result.inlier_rmse:.4f}")

    return np.array(result.transformation)


def transform_cameras(
    cameras: list[Camera],
    transform: NDArray[np.float64],
) -> list[Camera]:
    """
    Apply transformation to camera poses.

    Parameters
    ----------
    cameras : list of Camera
        Original cameras in COLMAP coordinates
    transform : ndarray of shape (4, 4)
        Transformation matrix (COLMAP -> LiDAR)

    Returns
    -------
    transformed_cameras : list of Camera
        Cameras with transformed poses
    """
    transformed = []
    R_transform = transform[:3, :3]
    t_transform = transform[:3, 3]

    for cam in cameras:
        # Get camera pose in world coordinates
        R_cam = cam.rotation_matrix
        t_cam = cam.tvec

        # Camera center in original coordinates
        center = -R_cam.T @ t_cam

        # Transform center to new coordinates
        new_center = R_transform @ center + t_transform

        # Transform rotation
        new_R = R_cam @ R_transform.T

        # Compute new tvec
        new_tvec = -new_R @ new_center

        # Convert rotation back to quaternion
        new_qvec = rotmat_to_qvec(new_R)

        transformed.append(Camera(
            id=cam.id,
            width=cam.width,
            height=cam.height,
            fx=cam.fx,
            fy=cam.fy,
            cx=cam.cx,
            cy=cam.cy,
            qvec=new_qvec,
            tvec=new_tvec,
            image_path=cam.image_path,
        ))

    return transformed


def qvec_to_rotmat(qvec: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])


def rotmat_to_qvec(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def initialize_from_lidar(
    las_path: Path,
    subsample: int | None = None,
) -> GaussianModel:
    """
    Initialize Gaussians from LiDAR point cloud.

    Parameters
    ----------
    las_path : Path
        Path to LAS/LAZ file
    subsample : int, optional
        If provided, subsample to this many points

    Returns
    -------
    GaussianModel
        Initial Gaussian parameters
    """
    import laspy

    las = laspy.read(str(las_path))
    points = np.vstack((las.x, las.y, las.z)).T

    # Get colors if available
    colors = None
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        # LAS stores colors as 16-bit, normalize to [0, 1]
        max_val = 65535 if las.red.max() > 255 else 255
        colors = np.vstack((
            las.red / max_val,
            las.green / max_val,
            las.blue / max_val,
        )).T

    # Subsample if requested
    if subsample is not None and len(points) > subsample:
        indices = np.random.choice(len(points), subsample, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]

    # Estimate initial scale from point density
    from scipy.spatial import cKDTree
    tree = cKDTree(points[:min(10000, len(points))])
    distances, _ = tree.query(points[:min(1000, len(points))], k=4)
    mean_nn_dist = np.mean(distances[:, 1:])  # Exclude self
    initial_scale = mean_nn_dist * 0.5

    print(f"  Estimated initial Gaussian scale: {initial_scale:.4f}")

    return GaussianModel.from_points(points, colors, initial_scale)


def get_training_device() -> str:
    """Determine the best available device for training."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def train_gaussians(
    model: GaussianModel,
    cameras: list[Camera],
    iterations: int = 30000,
    learning_rate: float = 0.001,
    densify_interval: int = 500,
    densify_until: int = 15000,
    device: str | None = None,
    checkpoint_interval: int = 5000,
    checkpoint_dir: Path | None = None,
) -> GaussianModel:
    """
    Train 3D Gaussian Splatting model.

    Parameters
    ----------
    model : GaussianModel
        Initial Gaussian parameters (from LiDAR)
    cameras : list of Camera
        Cameras with poses and image paths
    iterations : int
        Number of training iterations
    learning_rate : float
        Base learning rate
    densify_interval : int
        Densification check interval
    densify_until : int
        Stop densification after this iteration
    device : str, optional
        Training device (cuda, mps, cpu). Auto-detected if None.
    checkpoint_interval : int
        Save checkpoint every N iterations
    checkpoint_dir : Path, optional
        Directory for checkpoints

    Returns
    -------
    GaussianModel
        Trained model
    """
    try:
        import torch
        from gsplat import rasterization
    except ImportError as e:
        raise ImportError(
            "Training requires PyTorch and gsplat. Install with:\n"
            "  pip install torch gsplat\n"
            f"Error: {e}"
        )

    if device is None:
        device = get_training_device()

    print(f"\n  Training on device: {device}")
    print(f"  Initial Gaussians: {len(model.positions):,}")
    print(f"  Training images: {len(cameras)}")

    # Load all training images
    from PIL import Image

    train_images = []
    valid_cameras = []
    for cam in cameras:
        if cam.image_path and cam.image_path.exists():
            img = Image.open(cam.image_path).convert("RGB")
            img = img.resize((cam.width, cam.height))
            img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            train_images.append(img_tensor.to(device))
            valid_cameras.append(cam)

    if not train_images:
        raise ValueError("No valid training images found")

    print(f"  Loaded {len(train_images)} images")

    # Convert model to torch tensors
    positions = torch.tensor(model.positions, dtype=torch.float32, device=device, requires_grad=True)
    colors = torch.tensor(model.colors, dtype=torch.float32, device=device, requires_grad=True)
    opacities = torch.tensor(model.opacities, dtype=torch.float32, device=device, requires_grad=True)
    scales = torch.tensor(model.scales, dtype=torch.float32, device=device, requires_grad=True)
    rotations = torch.tensor(model.rotations, dtype=torch.float32, device=device, requires_grad=True)

    # Optimizer
    optimizer = torch.optim.Adam([
        {"params": [positions], "lr": learning_rate * 0.1},
        {"params": [colors], "lr": learning_rate},
        {"params": [opacities], "lr": learning_rate * 0.5},
        {"params": [scales], "lr": learning_rate * 0.1},
        {"params": [rotations], "lr": learning_rate * 0.1},
    ])

    # Training loop
    for iteration in range(iterations):
        # Random camera
        idx = np.random.randint(len(valid_cameras))
        cam = valid_cameras[idx]
        gt_image = train_images[idx]

        # Camera matrices
        K = torch.tensor(cam.intrinsic_matrix, dtype=torch.float32, device=device)
        R = torch.tensor(cam.rotation_matrix, dtype=torch.float32, device=device)
        t = torch.tensor(cam.tvec, dtype=torch.float32, device=device)

        # Build view matrix
        viewmat = torch.eye(4, device=device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t

        # Render
        try:
            rendered, alpha, info = rasterization(
                means=positions,
                quats=rotations,
                scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities),
                colors=colors,
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=cam.width,
                height=cam.height,
                near_plane=0.1,
                far_plane=1000.0,
            )
            rendered = rendered.squeeze(0)  # Remove batch dim
        except Exception as e:
            if iteration == 0:
                print(f"  Warning: Rasterization error: {e}")
            continue

        # Loss
        l1_loss = torch.abs(rendered - gt_image).mean()
        loss = l1_loss

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if iteration % 500 == 0:
            print(f"  Iteration {iteration:5d}: loss = {loss.item():.6f}")

        # Checkpoint
        if checkpoint_dir and iteration > 0 and iteration % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}.ply"
            checkpoint_model = GaussianModel(
                positions=positions.detach().cpu().numpy(),
                colors=colors.detach().cpu().numpy(),
                opacities=opacities.detach().cpu().numpy(),
                scales=scales.detach().cpu().numpy(),
                rotations=rotations.detach().cpu().numpy(),
            )
            export_gaussians_ply(checkpoint_model, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Return trained model
    return GaussianModel(
        positions=positions.detach().cpu().numpy(),
        colors=colors.detach().cpu().numpy(),
        opacities=opacities.detach().cpu().numpy(),
        scales=scales.detach().cpu().numpy(),
        rotations=rotations.detach().cpu().numpy(),
    )


def export_gaussians_ply(
    model: GaussianModel,
    output_path: Path,
) -> None:
    """
    Export Gaussians to PLY format compatible with standard 3DGS viewers.

    The PLY format follows the convention from the original 3DGS paper,
    with properties for position, color (as spherical harmonics), opacity,
    scale, and rotation.
    """
    n_points = len(model.positions)

    # Prepare spherical harmonics (just DC component for now)
    # DC = color * C0 where C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    sh_dc = model.colors / C0

    # Build PLY header
    header = f"""ply
format binary_little_endian 1.0
element vertex {n_points}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""

    with open(output_path, "wb") as f:
        f.write(header.encode("utf-8"))

        for i in range(n_points):
            # Position
            f.write(struct.pack("<fff",
                model.positions[i, 0],
                model.positions[i, 1],
                model.positions[i, 2],
            ))
            # Normals (placeholder)
            f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
            # SH DC coefficients (color)
            f.write(struct.pack("<fff",
                sh_dc[i, 0],
                sh_dc[i, 1],
                sh_dc[i, 2],
            ))
            # Opacity (already in logit space)
            f.write(struct.pack("<f", model.opacities[i]))
            # Scales (log space)
            f.write(struct.pack("<fff",
                model.scales[i, 0],
                model.scales[i, 1],
                model.scales[i, 2],
            ))
            # Rotation quaternion
            f.write(struct.pack("<ffff",
                model.rotations[i, 0],
                model.rotations[i, 1],
                model.rotations[i, 2],
                model.rotations[i, 3],
            ))


def export_initialization_ply(
    model: GaussianModel,
    output_path: Path,
) -> None:
    """
    Export initial Gaussians as simple PLY for visualization/debugging.

    This is a simpler format than the full 3DGS PLY, useful for checking
    the initialization before training.
    """
    n_points = len(model.positions)

    header = f"""ply
format ascii 1.0
element vertex {n_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

    with open(output_path, "w") as f:
        f.write(header)
        for i in range(n_points):
            x, y, z = model.positions[i]
            r = int(model.colors[i, 0] * 255)
            g = int(model.colors[i, 1] * 255)
            b = int(model.colors[i, 2] * 255)
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
