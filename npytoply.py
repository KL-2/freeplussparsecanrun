import numpy as np
import open3d as o3d

# 加载点云数据 (.npy 文件)
point_cloud_np = np.load("/home/user/software/freeplussparse/data/nerf_data/nerf_llff_data/111.npy")

threshold = 10  # 定义一个阈值

# 找出所有密度高于阈值的点
points = np.array(np.where(point_cloud_np > threshold)).T

# 将点坐标转换为浮点数
points = points.astype(np.float64)

# 创建Open3D点云对象
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 可视化点云
o3d.visualization.draw_geometries([point_cloud])

# # 确保数据类型是float64或float32
# point_cloud_np = point_cloud_np.astype(np.float64)
# print(point_cloud_np.shape)
# # 创建Open3D点云对象
# point_cloud = o3d.geometry.PointCloud()

# # 将NumPy数组转换为Open3D点云
# point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np)

# # 可视化点云
# o3d.visualization.draw_geometries([point_cloud])
# 保存点云为 PLY 格式
o3d.io.write_point_cloud(f"/home/user/software/freeplussparse/data/nerf_data/nerf_llff_data/output_point_cloud{threshold}.ply", point_cloud)