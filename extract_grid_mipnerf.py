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



import json

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 将 numpy 数组转换为列表
    elif hasattr(obj, '__dict__'):
        # 对象可能是一个自定义类实例
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    else:
        return obj
    
RESOLUTION = 256
def extract_fields(bound_min, bound_max, near, far, resolution, model, render_chunk_size):
    '''Creating point grid cube to extract density'''

    N = 128  # 每次处理的网格点数256
    print(resolution)
    
    # 创建均匀分布的网格点
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)#每个维度生成resolution个点，然后每N个一组,128/64=2
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
    
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)  # 存储密度值大小为resolution * resolution * resolution
    
    host_id = jax.host_id()
    for xi, xs in enumerate(X):#xi是索引，xs是元素
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                # print(len(xs), len(ys), len(zs))#N，N，N
                # print(xs)
                # input()
                # 生成网格点对应的坐标
                xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')#生成多维网格空间，xx,yy,zz为对应的坐标
                xx, yy, zz = xx.numpy(), yy.numpy(), zz.numpy()
                
                origins = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
                directions = np.zeros_like(origins)  # 假设方向为零，因为我们只关心密度
                viewdirs = np.zeros_like(origins)  # 同上
                radii = np.ones_like(origins[..., :1]) * 0.0008
                ones = np.ones_like(origins[..., :1])
                # print(f"origins shape: {origins.shape}")#(N*N*N,3)
                # input()
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
                        # print(rays.origins.shape[0])#2097152=4096*512#(N*N*N)
                        chunk_rays = (jax.tree_map(lambda r: r[i:i + render_chunk_size], rays))
                        # print(chunk_rays.origins.shape)#(1024, 3)
                        actual_chunk_size = chunk_rays.origins.shape[0]
                        rays_remaining = actual_chunk_size % jax.device_count()
                        if rays_remaining != 0:
                            padding = jax.device_count() - rays_remaining
                            chunk_rays = jax.tree_map(
                                lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays)
                            # print(chunk_rays.origins.shape)#(1024, 3)
                        else:
                            padding = 0 
                        # After padding the number of chunk_rays is always divisible by host_count.
                        rays_per_host = chunk_rays.origins.shape[0] // jax.host_count()
                        start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
                        chunk_rays = jax.tree_map(lambda r: utils.shard(r[start:stop]), chunk_rays)

                        # print(chunk_rays.origins.shape)#(1，RESOLUTION, 3)

                        chunk_renderings = model(None, chunk_rays)  # 模型返回两个字典组成的列表
                        # 2*(1, 1, render_chunk_size, N)

                        # print(f"chunk_renderings length: {len(chunk_renderings)}")
                        # for i, rendering in enumerate(chunk_renderings):
                        #     print(f"Chunk {i} keys and shapes:")
                        #     for k, v in rendering.items():
                        #         try:
                        #             print(f"  {k}: {v.shape}")
                        #         except AttributeError:
                        #             # 如果v不是数组而是其他类型，例如数值或列表，你可以使用len(v)来获取其大小
                        #             print(f"  {k}: length {len(v)}")
                        # input()

                        # Unshard the renderings
                        chunk_renderings = [{k: utils.unshard(v[0], padding)
                                            for k, v in r.items()}
                                            for r in chunk_renderings]
                        #2*(render_chunk_size,N)
                        # print(f"chunk_renderings length: {len(chunk_renderings)}")
                        # for i, rendering in enumerate(chunk_renderings):
                        #     print(f"Chunk {i} keys and shapes:")
                        #     for k, v in rendering.items():
                        #         try:
                        #             print(f"  {k}: {v.shape}")
                        #         except AttributeError:
                        #             # 如果v不是数组而是其他类型，例如数值或列表，你可以使用len(v)来获取其大小
                        #             print(f"  {k}: length {len(v)}")


                        densities = [d['density'] for d in chunk_renderings]  # 直接提取每个字典中的密度信息
                        # 假设 densities 已经正确初始化为你描述的形状：[2, (render_chunk_size, N)]

                        # 假设 densities 是一个形状为 [2, (render_chunk_size, N)] 的列表

                        # 第一步：对每个块内的 N 个值取平均，以获得 (render_chunk_size, 1) 的形状
                        density_means_per_block = [np.mean(block, axis=1, keepdims=True) for block in densities]

                        # 第二步：计算得到的两个 (render_chunk_size, 1) 形状数组的平均值
                        # 由于我们有两个这样的数组，我们可以直接对它们进行平均
                        # 注意：我们使用 np.mean 而不是 np.concatenate，因为我们需要计算的是平均值，而不是合并数组
                        final_density_mean = np.mean(np.array(density_means_per_block), axis=0)

                        # print(f"final_density_mean shape: {final_density_mean.shape}")      
                        # print(f"densities length: {len(densities)}")#2
                        # for i, density in enumerate(densities):
                        #     print(f"Density {i} shape: {density.shape}")
                        #（1024，128）
                        #2*（render_chunk_size，N）能跑（512，128）
                            
                        # density_mean = np.mean(densities, axis=0)  # 计算平均密度
                        # keys = [k for k in chunk_renderings[0] if k.find('density') == 0]
                        # for k in keys:
                        #     chunk_rendering[k] = [r[k] for r in chunk_renderings]

                        # print(f"density_mean shape: {density_mean.shape}")
                        #（render_chunk_size，N）

                        # 假设 densities 是一个形状为 [2, (render_chunk_size, N)] 的列表
                        # 下面的代码会对列表中的每个 (render_chunk_size, N) 形状的数组
                        # 沿着 N 维度计算平均值，以得到 (render_chunk_size, 1) 形状的数组
                        # 假设每个字典中的'density'键对应的值形状为(render_chunk_size, N)
                        # 首先将所有渲染块的密度值合并
                        # all_densities = np.concatenate([density for density in densities], axis=0)
                        # print(f"density_mean shape: {all_densities.shape}")
                        # # 然后计算这些值的平均密度
                        # density_mean = np.mean(all_densities, axis=0, keepdims=True).reshape(-1, 1)

                        # # 打印density_mean的形状
                        # print(f"density_mean shape: {density_mean.shape}")
                        # 计算每个块的平均密度并保持维度
                        # density_means = [np.mean(density, axis=1, keepdims=True) for density in densities]

                        # print("Shapes of elements in density_means:")
                        # for mean in density_means:
                        #     print(mean.shape)#(512,1)(render_chunk_size,1)

                        # 如果你需要将这些平均值合并为一个大数组，可以使用 np.concatenate
                        # density_mean_combined = np.concatenate(density_means, axis=0)

                        # 打印density_mean_combined的形状
                        # print(f"density_mean_combined shape: {density_mean_combined.shape}")#(1024,1)(render_chunk_size,1)

                        raws.append(final_density_mean)

                        # 打印raws中最新元素的形状
                        # print(f"Newest element shape in raws: {raws[-1].shape}")
                        # input()

                        # for i, raw in enumerate(raws):
                        #     print(f"raw {i} shape: {raw.shape}")#（render_chunk_size，N）（512，128）

                # print(f"raws length: {len(raws)}")#(2097152/render_chunk_size,render_chunk_size，N)=4096*512)
                #现在来看这个N就应该干掉
                ##raws:(N*N*N/render_chunk_size,render_chunk_size,N)
                # 合并块结果并重塑为网格形状
                sigma = np.concatenate(raws, axis=0)
                # print("sigma shape before max:", sigma.shape)  # 打印合并前 sigma 的形状
                sigma = np.maximum(sigma, 0)  # 确保密度值非负
                # print("sigma shape after max:", sigma.shape)  # 打印合并前 sigma 的形状
                # print(len(raws))
                # print(sigma.shape)
                val = sigma.reshape((len(xs), len(ys), len(zs)))#N，N，N
                # print("val shape:", val.shape)  # 打印val 的形状
                # input()
                u[xi * N:xi * N + len(xs), yi * N:yi * N + len(ys), zi * N:zi * N + len(zs)] = val
                # print(f"u shape: {u.shape}")


    return u


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
    # u = extract_fields(bound_min, bound_max, near, far, RESOLUTION, fn, config.render_chunk_size)
    print(f"Saved at {config.data_dir}111.npy")
    u = extract_fields(bound_min, bound_max, near, far, RESOLUTION, fn, 512)
    # import mcubes
    # N=256
    # threshold = 50.
    # print('fraction occupied', np.mean(sigma > threshold))
    # vertices, triangles = mcubes.marching_cubes(sigma, threshold)
    # # print('done', vertices.shape, triangles.shape)

    # ## Uncomment to save out the mesh
    # mcubes.export_mesh(vertices, triangles, "logs/lego_example/lego_{}.dae".format(N), "flower")

    # import trimesh

    # mesh = trimesh.Trimesh(vertices / N - .5, triangles)
    # mesh.show()

    # np.save(f"{FLAGS.data_dir}/{obj}.npy", u)
    
    np.save(f"{config.data_dir}111.npy", u)
    # print(f"Saved at {FLAGS.data_dir}/{obj}.npy")
    print(f"Saved at {config.data_dir}111.npy")
    

if __name__ == '__main__':
  app.run(extract_mesh)