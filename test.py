# import numpy as np

# # Given parameters
# Qw, Qx, Qy, Qz = 0.83854924346912851, 0.30202237074830207, -0.30862338299902498, -0.33221869500551066
# Tx, Ty, Tz = 3.4259354573311671, 0.078270005918451374, 2.2226539362842876
# fx, fy = 1439.7506076834127, 1439.7506076834127
# cx, cy = 320, 256

# # Calculate rotation matrix
# R = np.array([
#     [1 - 2*(Qy**2 + Qz**2), 2*(Qx*Qy - Qz*Qw), 2*(Qx*Qz + Qy*Qw)],
#     [2*(Qx*Qy + Qz*Qw), 1 - 2*(Qx**2 + Qz**2), 2*(Qy*Qz - Qx*Qw)],
#     [2*(Qx*Qz - Qy*Qw), 2*(Qy*Qz + Qx*Qw), 1 - 2*(Qx**2 + Qy**2)]
# ])

# # Invert rotation matrix to get camera coordinate to world coordinate
# R_inv = np.linalg.inv(R)

# # Translation vector
# T = np.array([Tx, Ty, Tz])

# # Camera center in world coordinate system
# Cw = -np.dot(R_inv, T)

# print("Camera center in world coordinates:", Cw)
import numpy as np

# Load camera parameters from cameras.txt
def load_camera_params(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        camera_params = {}
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            params = list(map(float, parts[4:]))
            camera_params[camera_id] = {'model': model, 'params': params}
    return camera_params

# Load image parameters from images.txt
def load_image_params(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        image_params = {}
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            image_params[image_id] = {'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz, 'tx': tx, 'ty': ty, 'tz': tz}
    return image_params

# Function to calculate camera center in world coordinates
def calculate_camera_center(camera_params, image_params, image_id):
    # Get camera parameters
    focal_length, principal_point_x, principal_point_y, radial_distortion = camera_params[1]['params']

    # Get image parameters
    qw, qx, qy, qz = image_params[image_id]['qw'], image_params[image_id]['qx'], image_params[image_id]['qy'], image_params[image_id]['qz']
    tx, ty, tz = image_params[image_id]['tx'], image_params[image_id]['ty'], image_params[image_id]['tz']

    # Calculate rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])

    # Invert rotation matrix to get camera coordinate to world coordinate
    R_inv = np.linalg.inv(R)

    # Translation vector
    T = np.array([tx, ty, tz])

    # Camera center in world coordinate system
    Cw = -np.dot(R_inv, T)

    return Cw

# Load camera parameters from cameras.txt
camera_params = load_camera_params("cameras.txt")

# Load image parameters from images.txt
image_params = load_image_params("images.txt")

# Calculate camera center in world coordinates for image ID 1
camera_center_world = calculate_camera_center(camera_params, image_params, 1)

print("Camera center in world coordinates:", camera_center_world)
