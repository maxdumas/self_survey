"""
Gaussian Splatting export command.

This command creates a trained 3D Gaussian Splatting model from a LiDAR
point cloud and a set of photos. The output can be viewed in standard
3DGS viewers for photorealistic novel view synthesis.

Workflow
--------

1. Run COLMAP on photos to estimate camera poses
2. Register COLMAP reconstruction to LiDAR coordinates via ICP
3. Initialize Gaussians from LiDAR points (dense, accurate geometry)
4. Train using differentiable rendering to match input photos
5. Export trained model as PLY

Prerequisites
-------------

- COLMAP must be installed (brew install colmap / apt install colmap)
- For training: PyTorch and gsplat (pip install torch gsplat)
- GPU recommended (CUDA or MPS) but CPU works (slowly)

Example Usage
-------------

    # Full pipeline: COLMAP + train + export
    preprocess splat merged_survey.laz --photos ./photos/ -o scene.ply

    # Just export initialization (skip training)
    preprocess splat merged_survey.laz --photos ./photos/ -o init.ply --skip-training

    # Use existing COLMAP workspace
    preprocess splat merged_survey.laz --colmap-workspace ./colmap/ -o scene.ply
"""

from pathlib import Path
from typing import Annotated

import cyclopts


def splat(
    input_file: Annotated[
        Path,
        cyclopts.Parameter(
            help="Input LAS/LAZ point cloud file",
        ),
    ],
    *,
    output: Annotated[
        Path,
        cyclopts.Parameter(
            "--output",
            "-o",
            help="Output PLY file path for trained Gaussians",
        ),
    ],
    photos: Annotated[
        Path | None,
        cyclopts.Parameter(
            "--photos",
            "-p",
            help="Directory containing input photos for training",
        ),
    ] = None,
    colmap_workspace: Annotated[
        Path | None,
        cyclopts.Parameter(
            "--colmap-workspace",
            help="Existing COLMAP workspace (skips COLMAP reconstruction)",
        ),
    ] = None,
    iterations: Annotated[
        int,
        cyclopts.Parameter(
            "--iterations",
            "-n",
            help="Number of training iterations",
        ),
    ] = 30000,
    skip_training: Annotated[
        bool,
        cyclopts.Parameter(
            "--skip-training",
            help="Skip training, just export LiDAR-initialized Gaussians",
        ),
    ] = False,
    subsample: Annotated[
        int | None,
        cyclopts.Parameter(
            "--subsample",
            help="Subsample LiDAR to this many points (for faster training)",
        ),
    ] = None,
    icp_distance: Annotated[
        float,
        cyclopts.Parameter(
            "--icp-distance",
            help="Max correspondence distance for COLMAP-to-LiDAR registration",
        ),
    ] = 50.0,
    checkpoint_interval: Annotated[
        int,
        cyclopts.Parameter(
            "--checkpoint-interval",
            help="Save checkpoint every N iterations (0 to disable)",
        ),
    ] = 5000,
    use_gpu: Annotated[
        bool,
        cyclopts.Parameter(
            "--gpu/--no-gpu",
            help="Use GPU for COLMAP feature extraction/matching",
        ),
    ] = True,
) -> None:
    """
    Export point cloud as trained 3D Gaussian Splat.

    This command creates a photorealistic 3D Gaussian Splatting model by
    combining LiDAR geometry with photo appearance. The LiDAR provides
    accurate initial geometry, while training optimizes appearance to
    match the input photos.

    The output PLY file can be viewed in standard 3DGS viewers like:
    - https://antimatter15.com/splat/
    - SuperSplat (https://playcanvas.com/supersplat/editor)
    - Luma AI viewer
    """
    import numpy as np
    import laspy

    from self_survey.gaussian_splatting import (
        check_colmap_installed,
        run_colmap,
        parse_colmap_text,
        register_colmap_to_lidar,
        transform_cameras,
        initialize_from_lidar,
        train_gaussians,
        export_gaussians_ply,
        export_initialization_ply,
        get_training_device,
    )

    # Validate inputs
    if not input_file.exists():
        raise cyclopts.ValidationError(f"Input file not found: {input_file}")

    if not skip_training:
        if photos is None and colmap_workspace is None:
            raise cyclopts.ValidationError(
                "Either --photos or --colmap-workspace is required for training.\n"
                "Use --skip-training to export initialization without training."
            )

    if photos is not None and not photos.exists():
        raise cyclopts.ValidationError(f"Photos directory not found: {photos}")

    if colmap_workspace is not None and not colmap_workspace.exists():
        raise cyclopts.ValidationError(f"COLMAP workspace not found: {colmap_workspace}")

    # Check dependencies
    if not skip_training or (photos is not None and colmap_workspace is None):
        if not check_colmap_installed():
            raise cyclopts.ValidationError(
                "COLMAP is not installed. Please install it:\n"
                "  macOS: brew install colmap\n"
                "  Ubuntu: sudo apt install colmap\n"
                "  Windows: Download from https://colmap.github.io/\n"
                "\nOr use --skip-training to export without camera poses."
            )

    # Step 1: Load LiDAR data
    print("=" * 60)
    print("STEP 1: Loading LiDAR point cloud")
    print("=" * 60)

    print(f"\nLoading {input_file}...")
    las = laspy.read(str(input_file))
    lidar_points = np.vstack((las.x, las.y, las.z)).T
    print(f"  Total points: {len(lidar_points):,}")
    print(f"  Bounds X: [{las.x.min():.2f}, {las.x.max():.2f}]")
    print(f"  Bounds Y: [{las.y.min():.2f}, {las.y.max():.2f}]")
    print(f"  Bounds Z: [{las.z.min():.2f}, {las.z.max():.2f}]")

    # Step 2: COLMAP reconstruction (if needed)
    cameras = None
    colmap_points = None

    if photos is not None and colmap_workspace is None:
        print("\n" + "=" * 60)
        print("STEP 2: Running COLMAP reconstruction")
        print("=" * 60)

        # Create workspace directory
        workspace_dir = output.parent / f"{output.stem}_colmap"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing photos from {photos}...")
        photo_count = len(list(photos.glob("*.jpg")) + list(photos.glob("*.JPG")) +
                         list(photos.glob("*.png")) + list(photos.glob("*.PNG")))
        print(f"  Found {photo_count} images")

        reconstruction = run_colmap(photos, workspace_dir, use_gpu=use_gpu)
        cameras = reconstruction.cameras
        colmap_points = reconstruction.points3d

        print(f"\n  Reconstructed {len(cameras)} cameras")
        print(f"  Sparse points: {len(colmap_points):,}")

    elif colmap_workspace is not None:
        print("\n" + "=" * 60)
        print("STEP 2: Loading existing COLMAP workspace")
        print("=" * 60)

        # Find sparse reconstruction
        text_dir = colmap_workspace / "sparse_text"
        if not text_dir.exists():
            # Try to find and convert binary format
            sparse_dirs = list((colmap_workspace / "sparse").iterdir()) if (colmap_workspace / "sparse").exists() else []
            if sparse_dirs:
                import subprocess
                text_dir = colmap_workspace / "sparse_text"
                text_dir.mkdir(exist_ok=True)
                subprocess.run([
                    "colmap", "model_converter",
                    "--input_path", str(sparse_dirs[0]),
                    "--output_path", str(text_dir),
                    "--output_type", "TXT",
                ], check=True, capture_output=True)

        if not text_dir.exists():
            raise cyclopts.ValidationError(
                f"Could not find COLMAP reconstruction in {colmap_workspace}"
            )

        # Determine image directory
        image_dir = colmap_workspace / "images"
        if not image_dir.exists():
            image_dir = photos if photos else colmap_workspace

        reconstruction = parse_colmap_text(text_dir, image_dir)
        cameras = reconstruction.cameras
        colmap_points = reconstruction.points3d

        print(f"  Loaded {len(cameras)} cameras")
        print(f"  Sparse points: {len(colmap_points):,}")

    # Step 3: Register COLMAP to LiDAR (if we have COLMAP data)
    transform = None
    if cameras is not None and colmap_points is not None and len(colmap_points) > 0:
        print("\n" + "=" * 60)
        print("STEP 3: Registering COLMAP to LiDAR coordinates")
        print("=" * 60)

        print("\nAligning coordinate systems via ICP...")
        transform = register_colmap_to_lidar(
            colmap_points,
            lidar_points,
            max_correspondence_distance=icp_distance,
        )

        # Apply transformation to cameras
        cameras = transform_cameras(cameras, transform)
        print(f"\n  Transformed {len(cameras)} camera poses")

        # Report transformation
        translation = transform[:3, 3]
        print(f"  Translation: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]")
    else:
        print("\n" + "=" * 60)
        print("STEP 3: Skipping registration (no COLMAP data)")
        print("=" * 60)

    # Step 4: Initialize Gaussians from LiDAR
    print("\n" + "=" * 60)
    print("STEP 4: Initializing Gaussians from LiDAR")
    print("=" * 60)

    print(f"\nCreating Gaussians from {len(lidar_points):,} points...")
    if subsample:
        print(f"  Subsampling to {subsample:,} points")

    model = initialize_from_lidar(input_file, subsample=subsample)
    print(f"  Initialized {len(model.positions):,} Gaussians")

    # Step 5: Training (if not skipped)
    if not skip_training and cameras is not None:
        print("\n" + "=" * 60)
        print("STEP 5: Training Gaussian Splatting model")
        print("=" * 60)

        # Check training dependencies
        try:
            import torch
            import gsplat
        except ImportError as e:
            print(f"\n  Warning: Training dependencies not available: {e}")
            print("  Install with: pip install torch gsplat")
            print("  Exporting initialization instead...")
            skip_training = True

        if not skip_training:
            device = get_training_device()
            print(f"\n  Device: {device}")
            print(f"  Iterations: {iterations:,}")

            checkpoint_dir = None
            if checkpoint_interval > 0:
                checkpoint_dir = output.parent / f"{output.stem}_checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Checkpoints: {checkpoint_dir}")

            model = train_gaussians(
                model,
                cameras,
                iterations=iterations,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
                device=device,
            )
    elif skip_training:
        print("\n" + "=" * 60)
        print("STEP 5: Skipping training (--skip-training)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("STEP 5: Skipping training (no camera data)")
        print("=" * 60)

    # Step 6: Export
    print("\n" + "=" * 60)
    print("STEP 6: Exporting Gaussian model")
    print("=" * 60)

    print(f"\nSaving to {output}...")
    output.parent.mkdir(parents=True, exist_ok=True)

    if skip_training or cameras is None:
        # Export simple initialization PLY
        export_initialization_ply(model, output)
        print("  Exported initialization PLY (simple format)")
    else:
        # Export full 3DGS PLY
        export_gaussians_ply(model, output)
        print("  Exported trained 3DGS PLY")

    file_size = output.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Gaussians: {len(model.positions):,}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput: {output}")

    if skip_training or cameras is None:
        print("\nTo view the point cloud:")
        print("  - CloudCompare, MeshLab, or any PLY viewer")
        print("\nTo train a full 3DGS model:")
        print(f"  preprocess splat {input_file} --photos <photo_dir> -o {output}")
    else:
        print("\nTo view the trained splat:")
        print("  - https://antimatter15.com/splat/")
        print("  - SuperSplat: https://playcanvas.com/supersplat/editor")
        print("  - Luma AI viewer")
