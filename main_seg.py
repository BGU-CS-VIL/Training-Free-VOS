from helper import *
from models_new import *
import matplotlib
import numpy as np
import torch
import argparse
import warnings
from tqdm import tqdm
import os
import gc

matplotlib.rcParams['savefig.pad_inches'] = 0
warnings.filterwarnings("ignore", category=DeprecationWarning)


def segment_video(args):
    print("Loading Features of ", args.scene_name)
    dir_images, dir_anno, dirs_un_anno = get_Davis_files(args.path, args.train_mode)
    features_scene = load_features_scene(dir_images, args.scene_name, args.path_feat)
    np_array_features = np.swapaxes(np.array(features_scene), 1, 2)
    array_feat_up = []

    # Increase resolution to avoid small clusters
    for i, feat in enumerate(np_array_features):
        upsampled_feat = torch.from_numpy(feat.T.reshape(args.n_features-0, 60, -1)).unsqueeze(0)
        upsampled_feat = F.interpolate(upsampled_feat, scale_factor=2, mode='bilinear', align_corners=True,
                                       recompute_scale_factor=True)[0]
        feat = upsampled_feat.numpy().reshape(args.n_features-0, -1).T
        array_feat_up.append(feat)
    np_array_features = np.array(array_feat_up)
    del array_feat_up
    n_patches = np_array_features.shape[1]
    num_patches = (60 * 2, n_patches // (60 * 2))
    im_size = (num_patches[0] * 4, num_patches[1] * 4)

    args.n_patches = n_patches
    args.num_patches = num_patches
    args.im_size = im_size

    xy_large2_= create_xy_feat_vmf((num_patches[0] // 2, num_patches[1] // 2))
    xy_large2 = create_xy_feat_vmf3((num_patches[0] // 2, num_patches[1] // 2))

    print("Loading Images")
    pre_images = preprocess_images(dir_images[args.scene_name])

    print("Loading Annotations")
    anno, size_original = get_annotation(dir_anno[args.scene_name])

    np_array_features = np.ascontiguousarray(np_array_features)
    print("Shape of features:  ", np_array_features.shape)

    args.num_items = len(np.unique(anno[0]))

    n_h = args.n_h
    n_w = args.n_w
    xy_init = xy_large2

    up_scale = 1

    first_feat = np_array_features[0]
    first_feat = torch.from_numpy(first_feat).cuda()
    color_feat = None
    args.crf_net = True
    if args.crf_net:
        Crf_net2 = ModelWithCrf(args.num_items, 5, kernel_size=5, potts=1.0, p_kernel=5).cuda()
        Crf_net4 = ModelWithCrf(args.num_items, 5, kernel_size=5, potts=1.0, p_kernel=5).cuda()

    first_anno = np.array(
        Image.fromarray(np.array(anno[0]).reshape(args.im_size[0], -1)).resize((num_patches[1], num_patches[0]),
                                                                               Image.NEAREST))
    first_anno = torch.from_numpy(first_anno).cuda()
    list_models = []
    objects_size = []
    for i in range(args.num_items):
        divider = 8
        objects_size.append(np.sqrt(((anno[0] == i).reshape(-1) + 0).sum()//divider))

    max_object_size = int((0+(np.array(objects_size).max())).clip(0,350))
    args.model_order = max_object_size
    args.ignore = []
    args.dict_lastseen = {}
    for i in range(args.num_items):
        args.dict_lastseen[str(i)+'_skip'] = 0
        if objects_size[i] <1.01:
            args.ignore.append(1)
            list_models.append(None)
            anno[0][anno[0]==i] =0
        else:
            if objects_size[i]<2:
                objects_size[i] = 16
            args.ignore.append(0)
            list_models.append(
                MixvMF3(x_dim=args.n_features + args.add_dim, order=args.model_order, num_models=args.num_models,
                            init_number=objects_size[i], background = (i == 0), num_feat=args.n_features).cuda().eval())

    feat_list, test_for_outliers, _ = get_features_permask_vmf_torch(
        (num_patches[0], num_patches[1]), first_feat,
        xy_large2 * args.xy_scale, first_anno,
        color=color_feat, args=args, masks_to_change=None)

    run_em_batch(list_models, feat_list, args, save_history=1, allow_new=False)
    num_iters = np_array_features.shape[0]
    pred = None
    starting_anno = None
    item_tensor = torch.arange(args.num_items).reshape(-1, 1, 1).cuda()
    os.makedirs(os.path.join(args.save_path, args.save_folder+'/') + str(args.scene_name) + '/', exist_ok=True)
    new_clustering_mask4 = None

    for ind in tqdm(range(0, num_iters)):
        ind_to_send = ind
        mask_all_to_change, mask_all_to_change2, mask_to_ignore, mask_to_keep, masks_to_ignore, masks_to_keep, = create_masks(
            args, first_anno, ind, n_h, n_w, pred,item_tensor)
        next_feat = np_array_features[ind_to_send]
        next_feat = torch.from_numpy(next_feat).cuda().float()

        next_feat_all = torch.cat((next_feat,xy_init),-1)

        next_feat_all = F.normalize(next_feat_all, dim=1, p=2).float()

        if ind == 0:
            new_clustering_mask4 = None
            mask_all_to_change2 = None
            list_of_ll_torch = None
        prob_objects, _, _, clusters_pred, feat_to_keep, list_of_ll_torch, _ = ll_compute_batch(
            list_models, next_feat_all, args, mask_all_to_change, masks_to_keep, masks_to_ignore, scale=1,
            gt_model_list=gt_model_list, init_scale=1, clusters_masks=new_clustering_mask4,
            mask_to_change_tight=mask_all_to_change2, list_of_ll_torch=list_of_ll_torch, first_anno=None)

        prob_objects_for_crf = upsample_prob_torch(prob_objects.max(1)[0].unsqueeze(0), args, scale=2)[0].argmax(0)
        prob_objects = prob_objects.max(1)[0].reshape(args.num_items, args.num_patches[0], -1)


        prob_object_for_outliers = torch.sort(prob_objects, dim=0, descending=True)[0][:2]
        prob_objects = prob_objects.argmax(0)

        threshold = args.std
        uncert_1 = torch.abs(prob_object_for_outliers[0] - prob_object_for_outliers[1])
        uncert_1[uncert_1 > threshold] = 255
        uncert_1[uncert_1 <= threshold] = 0
        uncert_1 = uncert_1.reshape(num_patches[0], num_patches[1])
        uncert_1 = uncert_1 == 255
        prob_objects_crf = prob_objects_for_crf.long()
        pred_crf_1 = torch.nn.functional.one_hot(prob_objects_crf.reshape(args.num_patches[0]*2, -1),
num_classes=args.num_items).permute(2, 0, 1).float()


        if args.crf_net:
            crf_scale = 2
            image = np.array(Image.fromarray(pre_images[ind].astype(np.uint8)).resize(
                (num_patches[1]  * crf_scale, num_patches[0] * crf_scale),
                Image.BICUBIC))
            if ind==0:
                anno_0 = np.array(Image.fromarray(anno[0]).resize(
                    (num_patches[1]  * crf_scale, num_patches[0] * crf_scale),
                    Image.NEAREST))
                anno_0 = torch.from_numpy(anno_0).cuda()
            image_for_Crf = torch.from_numpy(image).cuda()
            pred = crf_seg(Crf_net2,args,anno_0,image_for_Crf ,ind,pred_crf_1)
            pred_for_next = pred.mean(0).argmax(0)

            downsample_pred = F.interpolate(pred.float(), scale_factor=1 / 2,
                                                mode='bilinear', )

            pred = downsample_pred.mean(0).argmax(0)

        #####################################################################################################################3

        ####################################################################################################################


        uncert_1_send = uncert_1.reshape(-1)
        original_feat_list = None
        if ind == 0:
            pred = first_anno
        feat_list, test_for_outliers, xy_origin = get_features_permask_vmf_torch(
            (num_patches[0] * args.up_scale, num_patches[1] * args.up_scale), next_feat,
            xy_init * args.xy_scale, pred, uncert_1_send, original_feat_list,
            color=color_feat, args=args, masks_to_change=(pred.long() == prob_objects.long()),
            xy_origin_to_use=xy_large2_ * args.xy_scale)

        if ind > 0:
            run_em_batch(list_models, feat_list, args, get_history=1 + ind * args.time_to_send,
                         allow_new=(ind % 3 == 0), save_history=1 + ind * args.time_to_send,
                         use_knn=[new_clustering_mask4, xy_origin], masks_to_change=mask_all_to_change2)

        else:
            run_em_batch(list_models, feat_list, args, get_history=1 + ind * args.time_to_send,
                         allow_new=(ind % 3 == 0), save_history=1 + ind * args.time_to_send)

        if ind == 0:
            buffer_for_clusters_masks = 0
            new_clustering_mask4 = torch.zeros(
                (args.num_models, next_feat_all.shape[0], args.model_order * args.num_items)).cuda()

        prob_objects, max_objects, max_test_all, clusters_pred, _, _, new_clustering_mask4 = ll_compute_batch(
            list_models, next_feat_all, args, mask_all_to_change, masks_to_keep, masks_to_ignore,
            scale=args.up_scale, gt_model_list=gt_model_list, init_scale=up_scale,
            clusters_masks=new_clustering_mask4, mask_to_change_tight=mask_all_to_change2,
            list_of_ll_torch=list_of_ll_torch,
            buffer_for_clusters_masks=[buffer_for_clusters_masks, new_clustering_mask4], first_anno=starting_anno)

        starting_anno = None
        if args.mean:
            prob_objects = upsample_prob_torch(prob_objects.mean(1).unsqueeze(0), args, scale=2)
        else:
            prob_objects = upsample_prob_torch(prob_objects.max(1)[0].unsqueeze(0), args, scale=2)


        prob_objects = prob_objects.reshape(1, args.num_items, args.num_patches[0] * args.up_scale * 2, -1)
        prob_objects = prob_objects[0]

        prob_objects = prob_objects.argmax(0)
        new_clustering_mask4 = (new_clustering_mask4>0)+0

        crf_scale = 2

        pred_crf_2 = prob_objects.long()

        uncert_crf = torch.nn.functional.one_hot(
            pred_crf_2.reshape(args.num_patches[0] * crf_scale, -1),
            num_classes=args.num_items).permute(2, 0, 1).float()
        prev_uncert_crf = torch.nn.functional.one_hot(
            pred_for_next.reshape(args.num_patches[0]  * crf_scale, -1),
            num_classes=args.num_items).permute(2, 0, 1).float()

        if args.crf_net:
            pred = crf_seg(Crf_net4,args,anno_0, image_for_Crf,ind,uncert_crf * 1.0 + 0.0 * prev_uncert_crf)
            pred = upsample_crf_torch(pred, args, scale=4)
            pred = pred.mean(0).argmax(0)

        if ind == 0:
            pred = torch.from_numpy(np.array(anno[0])).cuda()

        #####################################################################################################################
        # Final
        ####################################################################################################################

        save_path = os.path.join(args.save_path, args.save_folder+'/') + str(args.scene_name) + '/' + str(ind).zfill(5) + '.png'

        save_final_result(args, ind, pre_images, pred, save_path, size_original)


    # Delete models

    for model in list_models:
        if model is not None:
            model.cpu()
            del model
    Crf_net2.cpu()
    del Crf_net2
    Crf_net4.cpu()
    del Crf_net4
    del list_models


def parse_arguments():
    parser = argparse.ArgumentParser(description="VMF-Seg")
    parser.add_argument("--loc", default=7, type=float, help="Location scale factor")
    parser.add_argument("--scene", default='all', help="Scene name")
    parser.add_argument("--time", default=0.33, type=float, help="Time factor")
    parser.add_argument("--reverse", default=0, type=int, help="Reverse scene order flag")
    parser.add_argument("--high_res", default=0, type=int, help="High resolution mode flag")
    parser.add_argument("--large_model", default=0, type=int, help="Use large model flag")
    parser.add_argument("--train_mode", default='val', choices=['train', 'val', 'test'], help="Training mode")
    parser.add_argument("--n_features", default=384, type=int, help="Number of features")
    parser.add_argument("--add_dim", default=64, type=int, help="Number of pe features")
    parser.add_argument("--max_iters", default=100, type=int, help="Maximum number of EM iterations")
    parser.add_argument("--rll_tol", default=1e-5, type=float, help="Tolerance of relative log-likelihood improvement")
    parser.add_argument("--num_models", default=10, type=int, help="Number of models")
    parser.add_argument("--std", default=1.0, type=float, help="Standard deviation for ood")
    parser.add_argument("--n_h", default=19, type=int, help="Height dimension for processing")
    parser.add_argument("--n_w", default=39, type=int, help="Width dimension for processing")
    parser.add_argument("--vis", action='store_true', help="Enable overlay visualization mode")
    return parser.parse_args()


if __name__ == '__main__':
    set_seed(123)
    args = parse_arguments()
    args.color_palette = np.load('./palette.npy')
    args.save_path = './'
    args.xy_scale = 0.25 * args.loc
    args.time_to_send = args.time
    args.dotxy_value = None
    args.up_scale = 1
    if args.train_mode=='val':
        args.save_folder = 'results_val'

    args.path = './data/DAVIS'
    args.file_list = './data/DAVIS/ImageSets/2017/val.txt'

    if args.scene == 'all':
        list_file = open(args.file_list, 'r')
        all_scenes = [line.rstrip('\n') for line in list_file.readlines()]
        all_scenes.sort()

    args.scene_name = args.scene
    args.path_feat = './features/'
    gt_model_list = None
    args.mean = False
    if args.scene_name == 'all':
        for scene in tqdm(all_scenes[:]):
            print("Current: ", scene)
            args.scene_name = scene
            args.max_iters = 100 
            try:
                set_seed(123)
                gc.collect()
                torch.cuda.empty_cache()
                segment_video(args)
            except Exception as E:
                print(E)
                print("Exception: ", E)
    else:
        segment_video(args)
