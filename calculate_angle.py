import numpy as np
# Load image parameters from images.txt
def load_image_params(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        image_params = {}
        for i in range(0, len(lines), 2):  # 从第二行开始处理特征点数据
            if lines[i].startswith('#'):
                continue #跳过以 # 开头的注释行
            line1 = lines[i].strip().split(' ')
            # line2 = lines[i + 1].strip().split(' ')
            image_id = int(line1[0])
            qw, qx, qy, qz = map(float, line1[1:5])
            tx, ty, tz = map(float, line1[5:8])
            image_params[image_id] = {qw,qx,qy,qz,tx,ty,tz}
    return image_params

def load_camera_params(file_path):
    """
    从相机位姿文件中读取相机的世界坐标信息。
    camera_id和world_position对应
    camera_world_positions[camera_id] = world_position
    Args:
        file_path: 相机位姿文件路径
    Returns:
        camera_world_positions: 字典,包含每个相机的ID和世界坐标信息
    这段代码假设相机位姿文件的每一行格式如下：
    相机ID, 模型, 宽度, 高度, PARAMS[]
    """
    # camera_world_positions = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        camera_params = {}
        for line in lines:
            if line.startswith('#'):
                continue#跳过注释
            values = line.strip().split()
            camera_id = int(values[0])
            # 世界坐标信息 (x, y, z)
            # world_position = [float(x) for x in values[5:8]]
            params = list(map(float, values[4:]))
            camera_params[camera_id] = params
            # camera_world_positions[camera_id] = world_position
    return camera_params

def calculate_camera_centers(camera_params):
    def calculate_camera_center(image_params):
        Qw, Qx, Qy, Qz, Tx, Ty, Tz = map(float, image_params)
        R = np.array([
            [1 - 2*(Qy**2 + Qz**2), 2*(Qx*Qy - Qz*Qw), 2*(Qx*Qz + Qy*Qw)],
            [2*(Qx*Qy + Qz*Qw), 1 - 2*(Qx**2 + Qz**2), 2*(Qy*Qz - Qx*Qw)],
            [2*(Qx*Qz - Qy*Qw), 2*(Qy*Qz + Qx*Qw), 1 - 2*(Qx**2 + Qy**2)]
        ])
        T = np.array([Tx, Ty, Tz])
        R_inv = np.linalg.inv(R)
        return -np.dot(R_inv, T)

    all_camera_centers = {}
    for image_id, image_params in camera_params.items():
        camera_center = calculate_camera_center(image_params)
        all_camera_centers[image_id] = camera_center
    
    return all_camera_centers

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
                        # print(f"angle{angle}")
                        # 查找当前相机对之间的最大夹角
                        max_angle = -float('inf')
                        for a in angles:
                            if a[1] == image_ids[i] and a[2] == image_ids[j] and a[3] > max_angle:
                                max_angle = a[3]
                        
                        # 如果当前夹角大于最大夹角，则添加到列表中
                        if angle > max_angle:
                            angles.append((point_id, image_ids[i], image_ids[j], angle))
    # print(f"angles{angles}")
    return angles

def readfromtxt(datasetbase):
    # 从相机位姿文件中读取相机的世界坐标信息
    camera_poses_file = datasetbase+'/cameras.txt'  # 替换为你的相机位姿文件路径
    images_file_path = datasetbase+'/images.txt'  # 替换为你的 images.txt 文件路径

    #改动
    camera_params = load_camera_params(camera_poses_file)
    # Load image parameters from images.txt
    image_params = load_image_params(images_file_path)
    # 使用示例
    camera_world_positions = calculate_camera_centers(image_params)
    # camera_world_positions = read_camera_poses(camera_poses_file)
    all_camera_centers = calculate_camera_centers(image_params)
    # for image_id, camera_center in all_camera_centers.items():
    #     print("Camera center for image ID {}: {}".format(image_id, camera_center))


    # 从 images.txt 文件中读取图像的特征点信息
    image_points = read_image_points(images_file_path)
    # print(f"image_points:{image_points}")
    # 从 points3D.txt 文件中读取三维点的坐标信息
    points3D_file_path = datasetbase+'/points3D.txt'  # 替换为你的 points3D.txt 文件路径
    point_coordinates = read_point_coordinates(points3D_file_path)

    # 汇总三维点的ID和图像的ID的相互联系
    point_image_relationship = summarize_point_image_relationship(image_points)

    # 获取关联图像数量最多的三维点的ID，并列出对应的相机ID
    sorted_point_image_relationship = sorted(point_image_relationship.items(), key=lambda x: len(x[1]), reverse=True)
    return point_image_relationship,sorted_point_image_relationship,camera_world_positions,point_coordinates

def auto_select_threshold(sorted_angles, max_percentile=50, step=1, min_drop_threshold=0.1):
    """
    自动选择阈值的功能。
    它接受一个已排序的夹角列表 sorted_angles,
    然后在给定的百分位数范围内尝试不同的阈值，
    选择平均夹角最大的阈值作为最佳阈值。

    尝试不同的百分位数，
    选择平均夹角最大的阈值作为最佳阈值。
    如果在搜索过程中出现平均角度的突变，
    则停止搜索并返回突变之前的平均值。

    Args:
        sorted_angles: 按照夹角大小排序后的角度列表。
        max_percentile: 最大的百分位数,取值范围为0到100。
        step: 每次增加百分位数的步长,默认为5。

    Returns:
        best_percentile: 最佳阈值对应的百分位数。
        best_score: 最佳阈值对应的平均夹角。
    """
    if not sorted_angles:
        print("sorted_angles is empty!")
        return None, None  # 返回空值
    
    # best_threshold = None
    best_score = float('-inf')
    best_percentile = None
    lowest_avg_angle = float('inf')  # 最低平均角度的初始值设为正无穷大
    # min_drop_threshold = 0.1  # 下降突变阈值为上一个百分位平均角度的10%
    record_pre_num_to_consider = 0
    for percentile in range(step, max_percentile + 1, step):
        # print(f"percentile is {percentile}")
        # threshold_index = int(percentile / 100 * len(sorted_angles))
        # threshold = sorted_angles[threshold_index]
        num_to_consider_pre = record_pre_num_to_consider
        
        num_to_consider = int(len(sorted_angles) * (float(percentile) / 100)) + 1
        record_pre_num_to_consider=num_to_consider

        if num_to_consider_pre==num_to_consider:
            selected_angles = sorted_angles[num_to_consider-1:num_to_consider]
            avg_angle = sum(selected[3] for selected in selected_angles) / 1
            # print(f"num_to_consider_pre==num_to_consider@@@@@@@@@@@@")
            
            # print(f"selected_angles is {selected_angles}")
            # print(f"avg_angle is {avg_angle}")

        else:
            selected_angles = sorted_angles[num_to_consider_pre:num_to_consider]
            avg_angle = sum(selected[3] for selected in selected_angles) / (num_to_consider-num_to_consider_pre)
            # print(f"avg_angle is {avg_angle}")
        
        if percentile==step:
            lowest_avg_angle=avg_angle
            # print(f"lowest_avg_angle is {lowest_avg_angle}")
        # 计算当前百分位数的平均角度与前1%平均角度的差值
        diff = lowest_avg_angle - avg_angle
        
        if diff > min_drop_threshold * lowest_avg_angle:  # 如果差值大于阈值
            # print(f"突变了")
            best_score = sum(selected[3] for selected in sorted_angles[:num_to_consider-step]) / (num_to_consider-step)
            best_percentile = percentile - step
            break  # 停止迭代
        
        best_score = sum(selected[3] for selected in sorted_angles[:num_to_consider]) / (num_to_consider)
        best_percentile = percentile
        
        # # 更新最低平均角度
        # if avg_angle < lowest_avg_angle:
        #     lowest_avg_angle = avg_angle

    return best_percentile, best_score

# def calculate_nearest_neighbors_average_angle(camera_world_positions, point_coordinates, point_image_relationship, allowed_cameras=None):
#     """
#     计算每个图像ID最近邻的两个相机之间的平均夹角，仅考虑allowed_cameras中指定的相机。
#     Args:
#         camera_world_positions: 字典，包含每个图像ID对应的相机中心世界坐标。
#         point_coordinates: 字典，包含每个三维点的坐标信息。
#         point_image_relationship: 字典，包含三维点ID和关联的图像ID列表。
#         allowed_cameras: 列表或None，如果提供，仅考虑列表中的图像ID。
#     Returns:
#         nearest_neighbors_average_angle: 字典，每个图像ID对应其最近邻的两个相机之间的平均夹角。
#     """
#     nearest_neighbors_average_angle = {}
#     if allowed_cameras is None:
#         allowed_cameras = list(camera_world_positions.keys())

#     for image_id in allowed_cameras:
#         camera_position = camera_world_positions.get(image_id)
#         if camera_position is None:
#             print(f"Warning: Camera ID {image_id} not found in camera_world_positions.")
#             continue

#         distances = {}
#         for other_id in allowed_cameras:
#             if other_id == image_id:
#                 continue  # Skip self

#             other_position = camera_world_positions.get(other_id)
#             if other_position is not None:
#                 distance = np.linalg.norm(np.array(camera_position) - np.array(other_position))
#                 distances[other_id] = distance
#             else:
#                 print(f"Warning: Camera ID {other_id} not found in camera_world_positions.")

#         if len(distances) < 2:
#             print(f"Warning: Not enough neighbors for camera ID {image_id} to calculate angles.")
#             continue

#         sorted_distances = sorted(distances.items(), key=lambda item: item[1])
       
#         # 选择最近的两个相机ID
#         if len(sorted_distances) >= 2:
#             nearest_neighbors_ids = [sorted_distances[0][0], sorted_distances[1][0]]
#         else:
#             # 如果没有足够的相机，则跳过当前相机
#             continue
        
#         # 计算与最近邻的两个相机观测到的所有三维点形成的夹角，并求平均值
#         angles = []
#         for point_id, images in point_image_relationship.items():
#             if image_id in images:
#                 for neighbor_id in nearest_neighbors_ids:
#                     if neighbor_id in images:
#                         point_pos = np.array(point_coordinates[point_id])
#                         angle = calculate_angle(np.array(camera_world_positions[image_id]),
#                                                 np.array(camera_world_positions[neighbor_id]),
#                                                 point_pos)
#                         angles.append(angle)
        
#         if angles:
#             average_angle = np.mean(angles)
#             nearest_neighbors_average_angle[image_id] = average_angle
    
#     return nearest_neighbors_average_angle
def calculate_nearest_neighbors_max_average_angle(camera_world_positions, point_coordinates, point_image_relationship, allowed_cameras=None):
    """
    计算每个图像ID最近邻的两个相机之间最大夹角的平均值，仅考虑allowed_cameras中指定的相机。
    Args:
        camera_world_positions: 字典，包含每个图像ID对应的相机中心世界坐标。
        point_coordinates: 字典，包含每个三维点的坐标信息。
        point_image_relationship: 字典，包含三维点ID和关联的图像ID列表。
        allowed_cameras: 列表或None，如果提供，仅考虑列表中的图像ID。
    Returns:
        nearest_neighbors_max_average_angle: 字典，每个图像ID对应其最近邻的两个相机之间最大夹角的平均值。
    """
    nearest_neighbors_max_average_angle = {}
    if allowed_cameras is None:
        allowed_cameras = list(camera_world_positions.keys())

    for image_id in allowed_cameras:
        camera_position = camera_world_positions.get(image_id)
        if camera_position is None:
            print(f"Warning: Camera ID {image_id} not found in camera_world_positions.")
            continue

        distances = {}
        for other_id in allowed_cameras:
            if other_id == image_id:
                continue  # Skip self

            other_position = camera_world_positions.get(other_id)
            if other_position is not None:
                distance = np.linalg.norm(np.array(camera_position) - np.array(other_position))
                distances[other_id] = distance
            else:
                print(f"Warning: Camera ID {other_id} not found in camera_world_positions.")

        if len(distances) < 2:
            print(f"Warning: Not enough neighbors for camera ID {image_id} to calculate angles.")
            continue

        sorted_distances = sorted(distances.items(), key=lambda item: item[1])
        
        # 选择最近的两个相机ID
        if len(sorted_distances) >= 2:
            nearest_neighbors_ids = [sorted_distances[0][0], sorted_distances[1][0]]
        else:
            continue
        
        # 计算与最近邻的两个相机观测到的所有三维点形成的最大夹角
        max_angles = []
        for point_id, images in point_image_relationship.items():
            if image_id in images:
                angles = [calculate_angle(np.array(camera_world_positions[image_id]),
                                          np.array(camera_world_positions[neighbor_id]),
                                          np.array(point_coordinates[point_id]))
                          for neighbor_id in nearest_neighbors_ids if neighbor_id in images]
                if angles:
                    max_angle = max(angles)
                    max_angles.append(max_angle)
        
        if max_angles:
            max_average_angle = np.mean(max_angles)
            nearest_neighbors_max_average_angle[image_id] = max_average_angle
    
    return nearest_neighbors_max_average_angle

def print_nearest_neighbors_average_angle(nearest_neighbors_average_angle):
    """
    打印每个图像ID最近邻的两个相机之间的平均夹角，并打印所有平均夹角的平均值。
    Args:
        nearest_neighbors_average_angle: 字典,每个图像ID对应其最近邻的两个相机之间的平均夹角。
    """
    total_angle = 0
    for image_id, angle in nearest_neighbors_average_angle.items():
        print(f"图像ID {image_id} 与其最近邻的两个相机之间的平均夹角为：{angle:.2f}度, file=f")
        total_angle += angle
    
    if nearest_neighbors_average_angle:
        average_of_averages = total_angle / len(nearest_neighbors_average_angle)
        print(f"所有平均夹角的平均值为：{average_of_averages:.3f}")
    else:
        print("没有可用的平均夹角数据。")


def main():
    with open('./output.txt', 'a') as f:
        use_allowed_cameras=True
    
        #   dataset_base,images_number,totalsparsity
        # dataset_info = [("flower", 34, 14),("flower", 34, 14)]
        # dataset_info = [("dtuscan9", 49, 14)]
        dataset_info = [("flower", 34, 14)]
        dataset_index_list=range(len(dataset_info))
    
        for dataset_index in list(dataset_index_list):   
            print(f"dataset_base is :{dataset_info[dataset_index][0]}", file=f)
            # 访问第一个元组中的值
            dataset_base = dataset_info[dataset_index][0]
            images_number = dataset_info[dataset_index][1]
            totalsparsity = dataset_info[dataset_index][2]#totalsparsity=14
    
            datasetbase=f'./{dataset_base}'
            sparsitylist=range(1, totalsparsity+1)
            # sparsitylist={1,2,3,4,5,6,7,8,9,10,11,12,13,14}
            # sparsitylist={12}
    
            point_image_relationship,sorted_point_image_relationship,camera_world_positions,point_coordinates\
                =readfromtxt(datasetbase)
    
            for sparsity in list(sparsitylist):
                # print(f"----------------------")
                print(f"sparsity is {sparsity}:", end=" ", file=f)
                # print(f"----------------------")
                # idx_sub = np.linspace(1, images_number-1,  images_number)[::sparsity]
                # idx_sub_list = idx_sub.astype(int).tolist()
                # allowed_cameras = idx_sub_list # 指定允许的相机列表
                # 使用 range 生成从1开始的整数序列
                allowed_cameras = list(range(1, images_number))[::sparsity]
    
                # allowed_cameras = [7, 9, 33]  # 指定允许的相机列表
                print(f"allowed_cameras:{allowed_cameras}", file=f)
    
                if use_allowed_cameras:
                    # angles = find_max_angles(sorted_point_image_relationship, camera_world_positions , point_coordinates,allowed_cameras)
                    # 示例使用：
                    # 假设 camera_world_positions, point_coordinates, point_image_relationship 已经由前面的代码计算得到
                    # nearest_neighbors_average_angle = calculate_nearest_neighbors_average_angle(camera_world_positions, point_coordinates, sorted_point_image_relationship,allowed_cameras)
                    nearest_neighbors_average_angle = calculate_nearest_neighbors_max_average_angle(camera_world_positions, point_coordinates, point_image_relationship, allowed_cameras)
    
                    print_nearest_neighbors_average_angle(nearest_neighbors_average_angle)
    
                else:
                    # angles = find_max_angles(sorted_point_image_relationship, camera_world_positions , point_coordinates)
                # print(f"angles:{angles[:5]}")
                # 对夹角进行排序，按照第四个元素（即夹角）的降序排列
                    # nearest_neighbors_average_angle = calculate_nearest_neighbors_average_angle(camera_world_positions, point_coordinates, sorted_point_image_relationship)
                    nearest_neighbors_average_angle = calculate_nearest_neighbors_max_average_angle(camera_world_positions, point_coordinates, point_image_relationship, allowed_cameras)
                    print_nearest_neighbors_average_angle(nearest_neighbors_average_angle)
      
           
                # sorted_angles = sorted(angles, key=lambda x: x[3], reverse=True)
                
                # # 输出结果1
                # # for i in range(output_count):
                # for i in range(1):
                #     point_id, image_id1, image_id2, angle = sorted_angles[i]
                #     # print(f"angle{angle}")
                #     print(f"夹角为：{angle} 度,三维点ID为 {point_id} 的点与图像ID为 {image_id1} 和图像ID为 {image_id2} 的相机之间")
                #     print(f"三维点ID为 {point_id}  ,坐标为 {[f'{coord:.3f}' for coord in point_coordinates[point_id]]}")
                #     print(f"图像1 ID为 {image_id1} ,坐标为 {[f'{coord:.3f}' for coord in camera_world_positions[image_id1]]}")
                #     print(f"图像2 ID为 {image_id2} ,坐标为 {[f'{coord:.3f}' for coord in camera_world_positions[image_id2]]}")#计算要输出的结果数量
    
                # # # 输出结果2
                # # # 计算前50%夹角的平均值
                # # output_count = int(len(sorted_angles) * 0.5)# 计算前50%的数量
                # # total_angle = sum(angle for _, _, _, angle in sorted_angles[:output_count])#占位符不会进行计算
                # # average_angle = total_angle / output_count
    
                # # print(f"前50%夹角的平均值为：{average_angle} 度")
    
                # # 输出结果3
                # # 自动确定阈值
                # percentile, avg_angle = auto_select_threshold(sorted_angles, 30, 1)
                # print(f"前{percentile}%夹角的平均值为：{avg_angle} 度")


if __name__ == '__main__':
    main()
