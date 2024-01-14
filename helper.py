import os
from glob import glob
import numpy as np
import pickle
import cv2
from PIL import Image
import torch
from palette_davis import palette as davis_palette
import imageio
import random
import math
import torch.nn.functional as F

def mask2rgb(mask, palette):
    mask_rgb = palette(mask)
    mask_rgb = mask_rgb[:,:,:3]
    return mask_rgb

def mask_overlay(mask, image, palette):
    """Creates an overlayed mask visualisation"""
    mask_rgb = mask2rgb(mask, palette)
    return 0.55 * image + 0.45 * mask_rgb*255

def create_2d_euclidean_rotary_emb(height, width, dim, device='cpu'):
    assert dim % 4 == 0, "The embedding dimension 'dim' should be divisible by 4."

    inv_freq = 1. / (10000 ** (torch.arange(0, dim, 4).float() / dim))
    inv_freq = inv_freq.to(device)

    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    y_coords, x_coords = y_coords.to(device).float(), x_coords.to(device).float()

    y_scaled = y_coords / height
    x_scaled = x_coords / width

    y_position = y_scaled.view(-1, 1) * inv_freq.unsqueeze(0)
    x_position = x_scaled.view(-1, 1) * inv_freq.unsqueeze(0)

    pos_emb = torch.stack([y_position.sin(), y_position.cos(), x_position.sin(), x_position.cos()], dim=-1)
    pos_emb = pos_emb.flatten(1, 2).view(height, width, -1)

    return pos_emb
def create_2d_rotary_positional_emb(height, width, dim, device='cpu'):
    assert dim % 4 == 0, "The embedding dimension 'dim' should be divisible by 4."

    inv_freq = 1. / (10000 ** (torch.arange(0, dim, 4).float() / dim))
    inv_freq = inv_freq.to(device)

    y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width))
    y_coords, x_coords = y_coords.to(device).float(), x_coords.to(device).float()

    y_position = y_coords.view(-1, 1) * inv_freq.unsqueeze(0)
    x_position = x_coords.view(-1, 1) * inv_freq.unsqueeze(0)

    pos_emb = torch.stack([y_position.sin(), y_position.cos(), x_position.sin(), x_position.cos()], dim=-1)
    pos_emb = pos_emb.flatten(1, 2).view(height, width, -1)

    return pos_emb

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def sinusoidal_embedding(d_model, height, width,steps_h=1,steps_w=1):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height//steps_h, width//steps_w)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width,steps_w).unsqueeze(1)
    pos_h = torch.arange(0., height,steps_h).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height//steps_h, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height//steps_h, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width//steps_w)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1,  width//steps_w)

    return pe.reshape(d_model*2,-1).T.numpy()

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] =':16:8'
    print(f"Random seed set as {seed}")



def load_features_scene(dir_images,scene,path_feat):
    all_files = []
    image_files = dir_images[scene]
    image_files.sort()
    for file in image_files[:]:
        path_to_load = file.replace('.jpg','.pkl').split('JPEGImages/480p')[-1]
        path_final = os.path.join(path_feat,path_to_load[1:])
        with open(path_final,'rb') as f:
            all_files.append(pickle.load(f))
    return np.array(all_files)



def get_Davis_files(path,train_mode):
    path_mode = os.path.join(path,'ImageSets/2017/')
    path_images = os.path.join(path,'JPEGImages/480p/')
    path_anno = os.path.join(path,'Annotations/480p/')

    if train_mode=='train':
        path = os.path.join(path_mode,'train.txt')
    elif train_mode =='val':
        path = os.path.join(path_mode,'val.txt')
    elif train_mode =='test':
        path = os.path.join(path_mode,'test-dev.txt')
    with open(path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    images_files = {}
    anno_files = {}
    un_anno_files = {}
    for line in lines:
        images_files[line]= glob(os.path.join(path_images,line,'*.jpg'))
        anno_files[line]= glob(os.path.join(path_anno,line,'*.png'))
        un_anno_files[line]= glob(os.path.join(path_anno,line,'*.png'))

    return images_files, anno_files,un_anno_files


def create_xy_feat_vmf3(num_patches, len_emb=64):
    xy_large2 = np.array(create_2d_rotary_positional_emb(num_patches[0]*2,num_patches[1]*2,len_emb).reshape(-1,len_emb))
    return torch.from_numpy(xy_large2).cuda()


def create_xy_feat_vmf(num_patches):
    xx = (np.linspace(0,1,num_patches[0]*2))
    yy = (np.linspace(0,1,num_patches[1]*2))
    pts = []
    for x in xx:
        for y in yy:
            pts.append([x,y])
    xy_large2 = np.array(pts, dtype = np.float32)

    return torch.from_numpy(xy_large2).cuda()


def preprocess_images(path):
    images = []
    path.sort()
    if type(path)==list:
        for p in path[:]:
            images.extend([read_frame(p)])
    else:
         images.extend([read_frame(path)])

    return images

def read_frame(frame_dir, scale_size=[480]):
    """
    read a single frame
    """
    img = np.array(Image.open(frame_dir))
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if(ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    return img

def get_annotation(path):
    anno = []
    path.sort()
    if type(path)==list:
        for i,p in enumerate(path[:]):
            anno.extend([read_seg(p)[0]])
            if i==0:
                size_original = read_seg(p)[-1]
    else:
         anno.extend([read_seg(path)[0]])
    return np.array(anno),size_original

def read_seg(seg_dir, scale_size=[480,480]):
    seg = Image.open(seg_dir)
    _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if(_w > _h):
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 64) * 64)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 64) * 64)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    seg = np.array(seg.resize((_tw , _th ), 0))
    return np.array(seg), (_w,_h)

def create_masks(args, first_anno, ind, n_h, n_w, pred,item_tensor):
    new_masks_to_keep = []
    new_masks_to_ignore = []
    if ind > 0:
        prev_anno = F.interpolate(pred.unsqueeze(0).unsqueeze(0).float(), size=(args.num_patches), mode='nearest')[
            0, 0].long()
    else:
        prev_anno = first_anno
    padded_prev_anno = torch.nn.functional.pad(prev_anno.float().unsqueeze(0).unsqueeze(0), (
        (args.n_w - 1) // 2, (args.n_w - 1) // 2, (args.n_h - 1) // 2, (args.n_h - 1) // 2), 'reflect')
    slided_window = \
        torch.nn.Unfold(kernel_size=(args.n_h, args.n_w), stride=1, )(padded_prev_anno).permute(0, 2, 1)[0].to(torch.int8)
    eq_tensor = []

    # Iterate through each window
    for item_index in range(item_tensor.shape[0]):
        has_match = (slided_window.to(torch.int8) == item_tensor[item_index].to(torch.int8)).any(-1).long()
        eq_tensor.append(has_match)

    eq_tensor = torch.stack(eq_tensor, dim=0)
    for i in range(args.num_items):
        if args.ignore[i]==1:
            new_masks_to_keep.append(mask_to_keep)
            new_masks_to_ignore.append(mask_to_ignore)
            continue
        slided_window_sum = eq_tensor[i]
        #In case object missing
        if slided_window_sum.sum()==0:
            slided_window_sum = args.dict_lastseen[str(i)]
            args.dict_lastseen[str(i)+'_skip'] = 1
        else:
            args.dict_lastseen[str(i)] = slided_window_sum
            args.dict_lastseen[str(i)+'_skip'] = 0

        mask_to_keep = slided_window_sum == n_h * n_w
        mask_to_ignore = slided_window_sum == 0

        new_masks_to_keep.append(mask_to_keep)
        new_masks_to_ignore.append(mask_to_ignore)
        mask_to_change = ((slided_window_sum < n_h * n_w) + 0 + (slided_window_sum > 0) + 0) > 0
        if i == 0:
            new_mask_all_to_change = mask_to_change
        else:
            new_mask_all_to_change = (new_mask_all_to_change + 0 + mask_to_change + 0) > 0


        mask_all_to_change = new_mask_all_to_change
        mask_all_to_change2 = new_mask_all_to_change
        masks_to_keep = new_masks_to_keep
        masks_to_ignore = new_masks_to_ignore
    return mask_all_to_change, mask_all_to_change2, mask_to_ignore, mask_to_keep, masks_to_ignore, masks_to_keep


def upsample_prob_torch(feat,args,scale):
    next_upsampled_feat = feat.permute(1,0,2).reshape(1,args.num_items,args.num_patches[0],-1)
    next_upsampled_feat = F.interpolate(next_upsampled_feat,scale_factor=scale, mode='bilinear', align_corners = False, recompute_scale_factor=False)
    next_upsampled_feat = next_upsampled_feat
    return next_upsampled_feat

def upsample_crf_torch(crf,args,scale=4):
    next_upsampled_feat = crf
    next_upsampled_feat = F.interpolate(next_upsampled_feat,scale_factor=2, mode='bilinear', align_corners = False, recompute_scale_factor=False)
    next_upsampled_feat = next_upsampled_feat
    return next_upsampled_feat

def save_final_result(args, ind, pre_images, pred, save_path, size_original):
    pred_to_send = Image.fromarray(pred.reshape(480, -1).cpu().numpy().astype(np.uint8)).resize(size_original, 0)
    frame_tar_seg = np.array(pred_to_send)
    imwrite_indexed(save_path, frame_tar_seg, args.color_palette)
    if args.vis:
        frame = np.array(Image.fromarray(pre_images[ind]).resize(size_original))
        overlay = mask_overlay(np.array(pred_to_send), frame, davis_palette)
        os.makedirs(os.path.join(args.save_path, 'results_overlay/') + str(args.scene_name) + '/', exist_ok=True)
        save_path = os.path.join(args.save_path, 'results_overlay/') + str(args.scene_name) + '/' + str(ind).zfill(
            5) + '.png'
        imageio.imwrite(save_path, (overlay).astype(np.uint8))

def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
      raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array.astype(np.uint8))
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def get_features_permask_vmf_torch(num_patches, feat, xy, labels ,distance_normal=None,original_feat_list=None,color=None,args=None,masks_to_change=None,xy_origin_to_use=None):

    labels_current = labels.reshape(-1)
    feat_norm=feat
    test_for_outliers = 0
    if color is None or not args.color:
        feat_norm = torch.cat((feat_norm,xy),1)
    else:
        feat_norm = torch.cat((feat_norm,color*args.color_scale,xy),1)
    feat_norm = F.normalize(feat_norm, dim=1, p=2)
    feat_norm_list = []
    dist_list = []

    for val in range(args.num_items):
        mask_hard_distance =(labels_current==val)
        if distance_normal is not None and 1:
            current_distance_normal = distance_normal
            mask_hard_distance =(((labels_current==val)+0) +(current_distance_normal))>1
            if masks_to_change is not None:
                mask_hard_distance = (mask_hard_distance+0+masks_to_change.reshape(-1)+0)>1



        current_feat_to_send_soft = feat_norm[mask_hard_distance]
        if xy_origin_to_use is None:
            xy_origin = xy[mask_hard_distance]
        else:
            xy_origin = xy_origin_to_use[mask_hard_distance]
        feat_norm_list.append(current_feat_to_send_soft)
        dist_list.append(xy_origin/args.xy_scale)

    return feat_norm_list,test_for_outliers,dist_list
