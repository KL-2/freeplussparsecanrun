import functools
from os import path

from absl import app
from absl import flags
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import numpy as np
import jax
from jax import random
import jax.numpy as jnp
# from internal import datasets
# from internal import math
# from internal import models
# from internal import utils
# from internal import vis
from internal import configs, datasets_depth, math, models, utils, vis  # pylint: disable=g-multiple-import

import torch
# import pdb

from tqdm import tqdm
# FLAGS = flags.FLAGS
# utils.define_common_flags()
configs.define_common_flags()
jax.config.parse_flags_with_absl()

# flags.DEFINE_bool(
#     'eval_once', True,
#     'If True, evaluate the model only once, otherwise keeping evaluating new'
#     'checkpoints if any exist.')
# flags.DEFINE_bool('save_output', True,
                #   'If True, save predicted images to disk.')

RESOLUTION = 128

import json

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 将 numpy 数组转换为列表
    elif hasattr(obj, '__dict__'):
        # 对象可能是一个自定义类实例
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    else:
        return obj

def extract_fields(bound_min, bound_max, near, far, resolution, model, render_chunk_size):
    '''Creating point grid cube to extract density'''
    # render_chunk_size *= 32  # 根据需要调整块大小#1024
    N = 64  # 每次处理的网格点数256
    print(resolution)
    
    # 创建均匀分布的网格点
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
    
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)  # 存储密度值
    host_id = jax.host_id()
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                # print(len(xs), len(ys), len(zs))
                # print(xs)
                # input()
                # 生成网格点对应的坐标
                xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                xx, yy, zz = xx.numpy(), yy.numpy(), zz.numpy()
                
                origins = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
                directions = np.zeros_like(origins)  # 假设方向为零，因为我们只关心密度
                viewdirs = np.zeros_like(origins)  # 同上
                radii = np.ones_like(origins[..., :1]) * 0.0008
                ones = np.ones_like(origins[..., :1])
                
                # 构造 Rays 对象，这里简化表示，具体结构需要根据 model 的要求调整
                rays = utils.Rays(
                    origins=origins,
                    directions=directions,
                    viewdirs=viewdirs,
                    radii=radii,
                    lossmult=ones,
                    near=ones * near,
                    far=ones * far)
              
                raws = []  # 存储每个块的处理结果
                # 分块处理
                with torch.no_grad():
                    for i in tqdm(range(0, rays.origins.shape[0], render_chunk_size)):
                        chunk_rays = (jax.tree_map(lambda r: r[i:i + render_chunk_size], rays))
                        # print(chunk_rays.origins.shape)#(1024, 3)
                        actual_chunk_size = chunk_rays.origins.shape[0]
                        rays_remaining = actual_chunk_size % jax.device_count()
                        if rays_remaining != 0:
                            padding = jax.device_count() - rays_remaining
                            chunk_rays = jax.tree_map(
                                lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays)
                            print(chunk_rays.origins.shape)#(1024, 3)
                        else:
                            padding = 0 
                        # After padding the number of chunk_rays is always divisible by host_count.
                        rays_per_host = chunk_rays.origins.shape[0] // jax.host_count()
                        start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
                        chunk_rays = jax.tree_map(lambda r: utils.shard(r[start:stop]), chunk_rays)

                        # serializable_chunk_rays = convert_to_serializable(chunk_rays)
                        # json_data = json.dumps(serializable_chunk_rays, indent=2)
                        # # 指定要写入的文件路径
                        # file_path = 'chunk_rays_output_extract_mesh.json'
                        # # 打开文件并写入 JSON 数据
                        # with open(file_path, 'w') as file:
                        #     file.write(json_data)
                        # # 通知用户文件已保存
                        # print(f"Data saved to {file_path}")
                        # raw = model(None, chunk_rays)
                        # # raws.append(np.mean(raw, axis=1))
                        # # raws.append(torch.mean(raw, dim=1).cpu())
                        # print("Keys in the first dictionary:", raw[0].keys())
                        # print("Keys in the second dictionary:", raw[1].keys())
                        # input()
                        # # raw = model(None, chunk_rays)[-1][-1].squeeze()
                        # # 打印第二个字典的键
                        # # print("Keys in the second dictionary:", ray_histories[1].keys())

                        # # 假设 chunk_output 是一个密度数组
                        # # raw = chunk_output.squeeze()  # 根据实际输出结构调整
                        # raws.append(raw)
                        raw = model(None, chunk_rays)  # 模型返回两个字典组成的列表
                        # 假设我们只关心密度信息
                        # densities = jax.tree_map(lambda d: d['density'], raw)  # 提取每个字典中的密度信息
                        densities = [d['density'] for d in raw]  # 直接提取每个字典中的密度信息
                        density_mean = np.mean(densities, axis=0)  # 计算平均密度
                        # 因为有两个字典，我们需要从每个字典中提取密度并进行合适的处理
                        # 这里简单地取平均值作为示例
                        # density_mean = np.mean([densities[0], densities[1]], axis=0)
                        # print(density_mean)
                        raws.append(density_mean)
                # 合并块结果并重塑为网格形状
                sigma = np.concatenate(raws, axis=0)
                sigma = np.maximum(sigma, 0)  # 确保密度值非负
                print(len(raws))
                print(sigma.shape)
                val = sigma.reshape((len(xs), len(ys), len(zs)))
                u[xi * N:xi * N + len(xs), yi * N:yi * N + len(ys), zi * N:zi * N + len(zs)] = val
    
    return sigma

# def extract_fields(bound_min, bound_max, near, far, resolution, model, config):
#     '''Creating point grid cube to extract density
#     Args:
#       bound_min: the minimun bound that the scene can reach (e.g. [-1,-1,-1])
#       bound_max: the maximun bound that the scene can reach (e.g. [1,1,1])
#       resolution:  is the number of distinct points in each dimension (x,y,z) 
#         that the point grid cube is compose.
#       query_func: function used for passing queries to network_fn.
#       fn: function. Model for predicting RGB and density at each point
#         in space.
#     Returns:
#       u: Estimated density per each point of the point grid cube. 
#     '''
#     N = 256
#     print(resolution)
#     X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
#     Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
#     Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

#     X = np.array(X)
#     Y = np.array(Y)
#     Z = np.array(Z)
#     u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
#     raws = []
    
#     for xi, xs in enumerate(X):
#         for yi, ys in enumerate(Y):
#             for zi, zs in enumerate(Z):
#                 xx, yy, zz = torch.meshgrid(xs, ys, zs)
#                 xx, yy, zz = np.array(xx), np.array(yy), np.array(zz)
#                 # print(xi, yi, zi)
#                 origins = np.concatenate([
#                     xx.reshape(-1, 1),
#                     yy.reshape(-1, 1),
#                     zz.reshape(-1, 1)
#                 ],
#                                 axis=-1)
#                 # origins = np.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3))
#                 # pdb.set_trace()
#                 directions = np.zeros_like(origins)
#                 viewdirs = np.zeros_like(origins)
#                 radii = np.ones_like(origins[..., :1]) * 0.0005
#                 ones = np.ones_like(origins[..., :1])
#                 rays = utils.Rays(
#                         origins=origins,
#                         directions=directions,
#                         viewdirs=viewdirs,
#                         radii=radii,
#                         lossmult=ones,
#                         near=ones * near,
#                         far=ones * far)
#                 # print('input:', origins.shape)
                
#                 for i in tqdm(range(0, rays[0].shape[0], config.chunk)):
#                     # print('chunk', i)
#                     # chunk_rays = namedtuple_map(lambda r: r[i:i + config.chunk].astype(np.float64), rays)
#                     chunk_rays = utils.namedtuple_map(lambda r: utils.shard(r[i:i + config.chunk]),
#                                       rays)
#                     # pdb.set_trace()
#                     raw = model(None, chunk_rays)[-1][-1].squeeze()
#                     raws.append(np.mean(raw, axis=1))

#                 # pdb.set_trace()
#                 sigma = np.concatenate(raws, axis=0)
#                 sigma = np.maximum((sigma), 0)
#                 # print('output:', sigma.shape)
#                 val =  sigma.reshape(len(xs), len(ys),len(zs))
#                 raws = []
#                 u[xi * N:xi * N + len(xs), 
#                   yi * N:yi * N + len(ys),
#                   zi * N:zi * N + len(zs)] = val
#     return u

import tensorflow as tf

def extract_mesh(unused_argv):

    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.experimental.set_visible_devices([], 'TPU')

    config = configs.load_config(save_config=False)
    if config.use_wandb:
        import wandb
        wandb.init(project=config.project, entity=config.entity, sync_tensorboard=True)
        wandb.run.name = config.expname
        wandb.run.save()
        wandb.config.update(config)
    if config.dataset_loader == 'dtu' or config.dataset_loader=='llff':  
        dataset = datasets_depth.load_dataset('test', config.data_dir, config)
    # elif config.dataset_loader == 'zed2':
    #   dataset = datasets_depth_zed.load_dataset('test', config.data_dir, config)
    # elif config.dataset_loader == 'kinect':
    #   dataset = datasets_depth_kinect.load_dataset('test', config.data_dir, config)
    # elif config.dataset_loader == 'iphone':
    #   dataset = datasets_depth_iphone.load_dataset('test', config.data_dir, config)
    else:
        print('not a defined dataset, please check it!')
        assert(False)

    model, init_variables = models.construct_mipnerf(
        random.PRNGKey(20200823),
        dataset.peek()['rays'],
        config)
    optimizer = flax.optim.Adam(config.lr_init).create(init_variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, init_variables



    ## ----- comment out as we use lpips instead ----- ##
    # print('WARNING: LPIPS calculation not supported. NaN values used instead.')
    # if config.eval_disable_lpips:
    #   lpips_fn = lambda x, y: np.nan
    # else:
    #   lpips_fn = lambda x, y: np.nan
    ## ------------------------------------------------ ##

    last_step = 0
    out_dir = path.join(config.checkpoint_dir,
                        'path_renders' if config.render_path else 'test_preds')
    path_fn = lambda x: path.join(out_dir, x)


    summary_writer = tensorboard.SummaryWriter(
            path.join(config.checkpoint_dir, 'eval'))
    # Fix for loading pre-trained models.
    try:
        # state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
        if config.dataset_loader == 'dtu':
            state = checkpoints.restore_checkpoint(config.checkpoint_dir, state, step=90000)
            print('########################## dtu')
        elif config.dataset_loader == 'llff':
            print('########################## llff')
            #state = checkpoints.restore_checkpoint(config.checkpoint_dir, state, step=90000)
            state = checkpoints.restore_checkpoint(config.checkpoint_dir, state, step=69768)
        else:
            print('########################## others')
            state = checkpoints.restore_checkpoint(config.checkpoint_dir, state, step=60000)

    except:  # pylint: disable=bare-except
        print('Using pre-trained model.')
        state_dict = checkpoints.restore_checkpoint(config.checkpoint_dir, None)
        for i in [9, 17]:
          del state_dict['optimizer']['target']['params']['MLP_0'][f'Dense_{i}']
        state_dict['optimizer']['target']['params']['MLP_0'][
            'Dense_9'] = state_dict['optimizer']['target']['params']['MLP_0'][
                'Dense_18']
        state_dict['optimizer']['target']['ppzer']['target']['params']['MLP_0'][
                'Dense_20']
        del state_dict['optimizerd']
        state = flax.serialization.from_state_dict(state, state_dict)

    step = int(state.optimizer.state.step)

    if config.freq_reg:
        # Rendering is forced to be deterministic even if training was randomized, as
        # this eliminates 'speckle' artifacts.
        freq_reg_mask = (
        math.get_freq_reg_mask(99, step, config.freq_reg_end, config.max_vis_freq_ratio),
        math.get_freq_reg_mask(27, step, config.freq_reg_end, config.max_vis_freq_ratio))
        def render_eval_fn(variables, _, rays):
            return jax.lax.all_gather(
                model.apply(
                    variables,
                    None,  # Deterministic.
                    rays,
                    resample_padding=config.resample_padding_final,
                    compute_extras=True,
                    freq_reg_mask=freq_reg_mask)[1], axis_name='batch')
        #返回的是ray_history
    else:
        def render_eval_fn(variables, _, rays):
            return jax.lax.all_gather(
                model.apply(
                    variables,
                    None,  # Deterministic.
                    rays,
                    resample_padding=config.resample_padding_final,
                    # compute_extras=True)[0], axis_name='batch')
                    compute_extras=True), axis_name='batch')

    # pmap over only the data input.
    render_eval_pfn = jax.pmap(
        render_eval_fn,
        in_axes=(None, None, 0),
        donate_argnums=2,
        # donate_argnums=(3,),
        axis_name='batch',
    )

    if step <= last_step:
        print(f'Checkpoint step {step} <= last step {last_step}, exit.')
        exit()
    print(f'Evaluating checkpoint at step {step}.')
    if config.eval_save_output and (not utils.isdir(out_dir)):
        utils.makedirs(out_dir)

    key = random.PRNGKey(0 if config.deterministic_showcase else step)
    perm = random.permutation(key, dataset.size)
    showcase_indices = np.sort(perm[:config.num_showcase_images])

    metrics = []
    showcases = []

    near = 2.0
    far = 6.0
    # obj = FLAGS.data_dir.split('/')[-1]
    obj = config.data_dir.split('/')[-1]
    
    if obj == 'hotdog':
        print('hotdog scale')
        xmin, xmax = [-1.5, 1.3]
        ymin, ymax = [-1.5, 1.2]
        zmin, zmax = [-1.2, 1.2]
    elif obj == 'mic':
        print('mic scale')
        xmin, xmax = [-1.5, 1.2]
        ymin, ymax = [-1.2, 1.2]
        zmin, zmax = [-1.2, 1.2]
    elif obj == 'ship':
        print('ship scale')
        xmin, xmax = [-1.5, 1.5]
        ymin, ymax = [-1.5, 1.5]
        zmin, zmax = [-1.2, 1.2]
    else:
        print('rest scale')
        xmin, xmax = [-1.2, 1.2]
        ymin, ymax = [-1.2, 1.2]
        zmin, zmax = [-1.2, 1.2]

    #调参

    bound_min = np.array([xmin, ymin, zmin])
    bound_max = np.array([xmax, ymax, zmax])
    fn = functools.partial(render_eval_pfn, state.optimizer.target)
    sigma = extract_fields(bound_min, bound_max, near, far, RESOLUTION, fn, config.render_chunk_size)
    
    import mcubes
    N=256
    threshold = 50.
    print('fraction occupied', np.mean(sigma > threshold))
    vertices, triangles = mcubes.marching_cubes(sigma, threshold)
    # print('done', vertices.shape, triangles.shape)

    ## Uncomment to save out the mesh
    mcubes.export_mesh(vertices, triangles, "logs/lego_example/lego_{}.dae".format(N), "flower")

    import trimesh

    mesh = trimesh.Trimesh(vertices / N - .5, triangles)
    mesh.show()

    # np.save(f"{FLAGS.data_dir}/{obj}.npy", u)
    
    np.save(f"{config.data_dir}/{obj}.npy", u)
    # print(f"Saved at {FLAGS.data_dir}/{obj}.npy")
    print(f"Saved at {config.data_dir}/{obj}.npy")
    

if __name__ == '__main__':
  app.run(extract_mesh)