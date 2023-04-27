import numpy as np

def export_textured_mesh(image, depth):
    pass


def show_depth_image():
    # This is a huge import so wait until it's called so the user has some feedback
    import open3d as o3d

    # There exists an EGBD image type in open3d
    image = o3d.io.read_image("all-in-focus.png")
    depth = o3d.io.read_image("depth.png")

    image_shape = np.asarray(image).shape

    depth_image = o3d.geometry.RGBDImage.create_from_color_and_depth( color=image,
                                                                     depth=depth,
                                                                     depth_scale=1/125500.0,
                                                                     depth_trunc=15500000.0,
                                                                     convert_rgb_to_intensity=False)
    cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic()
    cameraIntrinsics.set_intrinsics(image_shape[1], image_shape[0], 6000, 6000, image_shape[1] / 2, image_shape[0] / 2)

    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = cameraIntrinsics
    cam.extrinsic = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1, 0.],
                              [0., 0., 0., 1.]])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        image=depth_image,
        intrinsic=cam.intrinsic,
        extrinsic=cam.extrinsic)


    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    downsampled = pcd.random_down_sample(sampling_ratio=0.1)

    o3d.io.write_point_cloud('point_cloud.ply', downsampled)

    o3d.visualization.draw_geometries([downsampled])
