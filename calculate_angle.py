import numpy as np

def read_camera_poses(file_path):
    """
    从相机位姿文件中读取相机的世界坐标信息。
    camera_id和world_position对应
    camera_world_positions[camera_id] = world_position
    Args:
        file_path: 相机位姿文件路径
    Returns:
        camera_world_positions: 字典,包含每个相机的ID和世界坐标信息
    这段代码假设相机位姿文件的每一行格式如下：
    相机ID, 模型, 宽度, 高度, PARAMS[], x, y, z
    """
    camera_world_positions = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue#跳过注释
            values = line.strip().split()
            camera_id = int(values[0])
            # 世界坐标信息 (x, y, z)
            world_position = [float(x) for x in values[5:8]]
            camera_world_positions[camera_id] = world_position
    return camera_world_positions

def read_image_points(file_path):
    """
    从 images.txt 文件中读取图像的特征点信息。
    包括每个特征点的二维坐标 (X, Y) 和关联的三维点的ID (POINT3D_ID) 
    Args:
        file_path: images.txt 文件路径
    Returns:
        image_points: 字典，包含每个图像的特征点信息
    """
    
    image_points = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):  # 从第二行开始处理特征点数据
            if lines[i].startswith('#'):
                continue #跳过以 # 开头的注释行
            line1 = lines[i].strip().split(' ')
            line2 = lines[i + 1].strip().split(' ')
            image_id = int(line1[0])
            # print(f"image_id:{image_id}")#应该ok
            points_data = line2[0:]  # 获取特征点数据部分

            # 对于每两行中的特征点信息行，
            # 使用 strip() 方法去除首尾空格并使用 split(' ')
            #  方法按空格进行分割，得到特征点数据的列表。

            # print(f"points_data:{points_data}")
            points = []  # 存储特征点信息的列表
            for j in range(0, len(points_data), 3):  # 每四个元素表示一个特征点的信息
                x = float(points_data[j])
                # print(f"x:{points_data[j]}")

                y = float(points_data[j + 1])
                # print(f"y:{points_data[j + 1]}")

                if int(points_data[j + 2]) == -1:
                    continue  # 跳过没有关联到三维点的特征点
                point3d_id = int(points_data[j + 2])
                # print(f"points_data[j + 2]:{points_data[j + 2]}")

                points.append((x, y, point3d_id))
            image_points[image_id] = points
        # print("image_points:",image_points)
    return image_points

# # 测试读取特征点信息
# file_path = 'images.txt'  # 你的文件路径
# image_points = read_image_points(file_path)
# print(image_points)




def read_point_coordinates(file_path):
    """
    从 points3D.txt 文件中读取三维点的坐标信息。
    这段代码将输出每个三维点ID和关联的图像ID列表
    表示它们之间的相互联系。
    # 3D point list with one line of data per point:
    #POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] 
    as (IMAGE_ID, POINT2D_IDX)
    Args:
        file_path: points3D.txt 文件路径
    Returns:
        point_coordinates: 字典，包含每个三维点的坐标信息
    """
    point_coordinates = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            values = line.strip().split()
            point_id = int(values[0])
            coordinates = [float(value) for value in values[1:4]]
            point_coordinates[point_id] = coordinates
    return point_coordinates

def summarize_point_image_relationship(image_points):
    """
    汇总三维点的ID和图像的ID的相互联系。
    Args:
        image_points: 字典，包含每个图像的特征点信息
    Returns:
        point_image_relationship: 字典,包含三维点ID和关联的图像ID列表
    """
    point_image_relationship = {}
    for image_id, points in image_points.items():
        for point in points:
            point_id = point[2]  # 获取三维点的ID
            if point_id not in point_image_relationship:
                point_image_relationship[point_id] = []
            point_image_relationship[point_id].append(image_id)
    return point_image_relationship

def calculate_angle(camera1_position, camera2_position, point_position):
    """
    计算两个相机观测同一三维点时的夹角。
    接受两个相机的位置信息作为输入，并计算这两个相机观测同一三维点时的夹角。
    Args:
        camera1_position: 第一个相机在世界坐标系中的位置
        camera2_position: 第二个相机在世界坐标系中的位置
        point_position: 三维点在世界坐标系中的位置
    Returns:
        angle: 夹角（弧度）
    """
    vector_camera1_to_point = point_position - camera1_position
    vector_camera2_to_point = point_position - camera2_position
    cos_angle = np.dot(vector_camera1_to_point, vector_camera2_to_point) / (np.linalg.norm(vector_camera1_to_point) * np.linalg.norm(vector_camera2_to_point))
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)
    return angle_degrees

def find_max_angles(sorted_point_image_relationship, camera_world_positions,point_coordinates,allowed_cameras=None):
    angles = []
    for point_id, image_ids in sorted_point_image_relationship:
        # 获取三维点的世界坐标
        point_position = np.array(point_coordinates[point_id])
        
        # 获取每个相机的世界坐标
        camera_positions = [np.array(camera_world_positions[image_id]) for image_id in image_ids]
        
        # 计算每个不同相机对与三维点世界坐标连线之间的夹角，并将夹角保存到列表中
        for i in range(len(camera_positions)):
            for j in range(i + 1, len(camera_positions)):
                if allowed_cameras is None or (image_ids[i] in allowed_cameras and image_ids[j] in allowed_cameras):
                    if image_ids[i] != image_ids[j]:  # 检查相机是否相同
                        angle = calculate_angle(camera_positions[i], camera_positions[j], point_position)
                        
                        # 查找当前相机对之间的最大夹角
                        max_angle = -float('inf')
                        for a in angles:
                            if a[1] == image_ids[i] and a[2] == image_ids[j] and a[3] > max_angle:
                                max_angle = a[3]
                        
                        # 如果当前夹角大于最大夹角，则添加到列表中
                        if angle > max_angle:
                            angles.append((point_id, image_ids[i], image_ids[j], angle))
    return angles

def readfromtxt(datasetbase):
    # 从相机位姿文件中读取相机的世界坐标信息
    camera_poses_file = datasetbase+'/cameras.txt'  # 替换为你的相机位姿文件路径
    camera_world_positions = read_camera_poses(camera_poses_file)

    # 从 images.txt 文件中读取图像的特征点信息
    images_file_path = datasetbase+'./images.txt'  # 替换为你的 images.txt 文件路径
    image_points = read_image_points(images_file_path)
    # print(f"image_points:{image_points}")
    # 从 points3D.txt 文件中读取三维点的坐标信息
    points3D_file_path = datasetbase+'./points3D.txt'  # 替换为你的 points3D.txt 文件路径
    point_coordinates = read_point_coordinates(points3D_file_path)

    # 汇总三维点的ID和图像的ID的相互联系
    point_image_relationship = summarize_point_image_relationship(image_points)

    # 获取关联图像数量最多的三维点的ID，并列出对应的相机ID
    sorted_point_image_relationship = sorted(point_image_relationship.items(), key=lambda x: len(x[1]), reverse=True)
    return sorted_point_image_relationship,camera_world_positions,point_coordinates


def main():
    use_allowed_cameras=True

    #   dataset_base,images_number,totalsparsity
    # dataset_info = [("flower", 34, 14),("flower", 34, 14)]
    dataset_info = [("flower", 34, 14)]
    dataset_index_list=range(len(dataset_info))
    for dataset_index in list(dataset_index_list):   
        print(f"dataset_base is :{dataset_info[dataset_index][0]}")
        # 访问第一个元组中的值
        dataset_base = dataset_info[dataset_index][0]
        images_number = dataset_info[dataset_index][1]
        totalsparsity = dataset_info[dataset_index][2]

        datasetbase=f'./{dataset_base}'
        # totalsparsity=14
        # sparsitylist=range(1, totalsparsity+1)
        # sparsitylist={1,2,3,4,5,6,7,8,9,10,11,12,13,14}
        sparsitylist={1}

        sorted_point_image_relationship,camera_world_positions,point_coordinates=readfromtxt(datasetbase)

        for sparsity in list(sparsitylist):
            print(f"----------------------")
            print(f"sparsity is {sparsity}")
            print(f"----------------------")
            idx_sub = np.linspace(0, images_number - 1,  images_number)[::sparsity]
            idx_sub_list = idx_sub.astype(int).tolist()
            allowed_cameras = idx_sub_list # 指定允许的相机列表
            # allowed_cameras = [7, 9, 33]  # 指定允许的相机列表
            print(f"allowed_cameras:{allowed_cameras}")

            if use_allowed_cameras:
                angles = find_max_angles(sorted_point_image_relationship, camera_world_positions , point_coordinates,allowed_cameras)
            else:
                angles = find_max_angles(sorted_point_image_relationship, camera_world_positions , point_coordinates)

            # 对夹角进行排序，按照第四个元素（即夹角）的降序排列
            sorted_angles = sorted(angles, key=lambda x: x[3], reverse=True)

            # 输出结果
            # 计算要输出的结果数量
            output_count = int(len(sorted_angles) * 0.1)

            # 输出结果
            # for i in range(output_count):
            for i in range(5):
                point_id, image_id1, image_id2, angle = sorted_angles[i]
                print(f"三维点ID为 {point_id} 的点与图像ID为 {image_id1} 和图像ID为 {image_id2} 的相机之间的夹角为：{np.degrees(angle)} 度")
                print(f"三维点ID为 {point_id}  ,坐标为 {[f'{coord:.3f}' for coord in point_coordinates[point_id]]}")
                print(f"图像1 ID为 {image_id1} ,坐标为 {[f'{coord:.3f}' for coord in camera_world_positions[image_id1]]}")
                print(f"图像2 ID为 {image_id2} ,坐标为 {[f'{coord:.3f}' for coord in camera_world_positions[image_id2]]}")

if __name__ == '__main__':
    main()