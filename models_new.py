import numpy as np
import utils
import pacnet.paccrf as paccrf
import copy
import torch as th
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import return_norm


def kernel_func(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    for i in range(len(X2.shape) - 2):
        X1 = torch.fron_numpy(np.array([X1])).unsqueeze(0).cuda()
    sqdist = torch.sqrt(X1 ** 2 + X2 ** 2 - 2 * X1 * X2)
    time = sigma_f ** 2 * torch.exp(-0.5 / l ** 2 * sqdist)
    time[X2 == 0] = 0
    return time


def k_means_pp_init(X, K,num_feat=384,num_models=10):

    if X.shape[0]<2:
        means = [X[0].unsqueeze(0).repeat(num_models,1)]
    else:
        #means = [X[torch.linspace(1,X.shape[0]-1,num_models).long()]]
        # Select the first mean randomly
        means = [X[torch.randint(1, X.shape[0]-1, (num_models,))]]
    for k in range(1, K):

        next_mean_idx = torch.matmul(return_norm(X[:,  num_feat+3:]).unsqueeze(0), return_norm(torch.stack(means[:])[:, :,  num_feat+3:]).permute(0, 2, 1)).sum(0).argmin(0)

        means.append(X[next_mean_idx])


    return torch.stack(means)


class ModelWithCrf(nn.Module):
    def __init__(self, num_classes=2, num_steps=5, pairwise=('4d_5_16_1',), loose=True,
                 use_yx=True, shared_scales=False, adaptive_init=True, kernel_size=7, potts=6.0, p_kernel=9):

        super(ModelWithCrf, self).__init__()
        self.num_classes = num_classes
        self.use_yx = use_yx
        self.shared_scales = shared_scales
        if isinstance(pairwise, str):
            pw_strs = []
            for s in pairwise.split('p')[1:]:
                l_ = 3 if s[2] == 's' else 2
                pw_strs.append('_'.join((s[:l_], s[l_], s[(l_ + 1):-1], s[-1])))
        else:
            pw_strs = pairwise
        crf_params = dict(num_steps=num_steps,
                          perturbed_init=True,
                          fixed_weighting=False,
                          unary_weight=0.85,
                          pairwise_kernels=[])
        for pw_str in pw_strs:
            for j in range(1):
                t_, k_, d_, b_ = pw_str.split('_')
                pairwise_param = dict(kernel_size=kernel_size,
                                      dilation=64,
                                      blur=1,
                                      compat_type=('potts' if t_.startswith('0d') else t_[:2]),
                                      spatial_filter=t_.endswith('s'),
                                      # spatial_filter=True,
                                      pairwise_weight=1)
            crf_params['pairwise_kernels'].append(pairwise_param)
            for j in range(1):
                t_, k_, d_, b_ = pw_str.split('_')
                pairwise_param = dict(kernel_size=kernel_size,
                                      dilation=16,
                                      blur=1,
                                      compat_type=('potts' if t_.startswith('0d') else t_[:2]),
                                      spatial_filter=t_.endswith('s'),
                                      # spatial_filter=True,
                                      pairwise_weight=1)
            crf_params['pairwise_kernels'].append(pairwise_param)
            for j in range(1):
                t_, k_, d_, b_ = pw_str.split('_')
                pairwise_param = dict(kernel_size=p_kernel,
                                      dilation=1,
                                      blur=1,
                                      compat_type='potts',
                                      spatial_filter=True,
                                      pairwise_weight=potts)
                crf_params['pairwise_kernels'].append(pairwise_param)

        CRF = paccrf.PacCRFLoose if loose else paccrf.PacCRF
        self.crf = CRF(self.num_classes, **crf_params)
        self.feat_scales = nn.ParameterList()
        self.potts = potts
        self.potts_w = nn.Parameter(torch.tensor([1.0]))
        for s in pw_strs:
            for j in range(3):
                fs, dilation = float(s.split('_')[1]), float(s.split('_')[2])
                p_sc = (((fs - 1) * dilation + 1) / 4.0) if adaptive_init else 100.0
                c_sc = 30.0
                if use_yx:
                    scales = th.tensor([p_sc, c_sc] if shared_scales else ([p_sc] * 2 + [c_sc] * 3), dtype=th.float32)
                else:
                    scales = th.tensor(c_sc if shared_scales else [c_sc] * 3, dtype=th.float32)
                self.feat_scales.append(nn.Parameter(scales))

    def forward(self, unary, x, out_crop=None, create_position=False, use_preposition=False):
        in_h, in_w = x.shape[2:]
        if create_position:
            self.pos_feat = []
        if use_preposition:
            preposition = self.pos_feat
        else:
            preposition = [None, None, None]
        if out_crop is not None and (out_crop[0] != in_h or out_crop[1] != in_w):
            x = x[:, :, :out_crop[0], :out_crop[1]]
        if self.use_yx:
            if self.shared_scales:
                edge_feat = [paccrf.create_YXRGB(x, yx_scale=sc[0], rgb_scale=sc[1], create_position=create_position,
                                                 use_preposition=preposition) for ind, sc in
                             enumerate(self.feat_scales)]
            else:
                edge_feat = [
                    paccrf.create_YXRGB(x, scales=sc, create_position=create_position, use_preposition=preposition[ind])
                    for ind, sc in enumerate(self.feat_scales)]

            if create_position:
                for i in range(len(self.feat_scales)):
                    self.pos_feat.append(edge_feat[i][1])
                edge_feat = [feat[0] for feat in edge_feat]
        else:
            edge_feat = [x * (1.0 / rgb_scale.view(-1, 1, 1)) for rgb_scale in self.feat_scales]

        if create_position:
            self.pos_feat.append(paccrf.create_position_feats(x.shape[2:], self.potts, bs=x.shape[0], device=x.device))

        if use_preposition:
            edge_feat.append(self.pos_feat[-1])
        else:
            edge_feat.append(paccrf.create_position_feats(x.shape[2:],self.potts_w, bs=x.shape[0], device=x.device))
        unary = unary * 5
        out = self.crf(unary, edge_feat)

        return out


class MixvMF3(nn.Module):
    '''
    MixvMF(x) = \sum_{m=1}^M \alpha_m vMF(x; mu_m, kappa_m)
    '''

    def __init__(self, x_dim, order, num_models=1, reg=1e-6, init_number=5, background=False, num_feat=384):

        super(MixvMF3, self).__init__()
        self.history_count = 20

        self.x_dim = x_dim
        self.order = order
        self.reg = reg
        self.alpha_logit = (0.01 * torch.randn(num_models, order)).cuda() * 0 + 1.1
        self.mu_unnorm = (torch.randn((num_models, order, x_dim))).cuda()
        self.logkappa = (0.01 * torch.randn((num_models, order))).cuda()
        self.history = torch.zeros((num_models, self.history_count, order, x_dim)).cuda()
        self.history_time = torch.zeros((num_models, self.history_count)).cuda()
        self.history_q = torch.zeros((num_models, self.history_count, order)).cuda()
        self.ind = 0
        self.num_models = num_models
        self.mask_alpha = torch.ones((num_models, order)).cuda().requires_grad_(False)
        self.num_feat = num_feat
        if background:
            max_number = int(init_number)
            max_number = np.sqrt(max_number)
            list_of_k = (np.linspace(7.5, max_number, self.num_models) ** 2).astype(np.int32)

            for k in range(self.num_models):
                self.mask_alpha[k, int(list_of_k[k]):] = 0
                self.alpha_logit[k, int(list_of_k[k]):] = -99999
        else:
            max_number = int(init_number)
            max_number = np.sqrt(max_number)
            list_of_k = (np.linspace(1.5, max_number, self.num_models) ** 2).astype(np.int32)
            for k in range(self.num_models):
                max_number = int(max_number)
                self.mask_alpha[k, int(list_of_k[k]):] = 0
                self.alpha_logit[k, int(list_of_k[k]):] = -99999

    def set_mask(self, masks):
        self.mask_alpha = masks

    def set_params(self, alpha, mus, kappas):

        with torch.no_grad():
            self.alpha_logit = (torch.log(alpha + utils.realmin))
            self.mu_unnorm = (mus)
            self.logkappa = (torch.log(kappas + utils.realmin))

    def logcmkappox(d, z):
        v = d / 2 - 1
        return torch.sqrt((v + 1) * (v + 1) + z * z) - (v - 1) * torch.log(
            v - 1 + torch.sqrt((v + 1) * (v + 1) + z * z))

    def open_new_cluster(self, threshold, feats):
        min_zeros_value = self.mask_alpha.argmin(-1)

        threshold = ((threshold + 0) + (min_zeros_value <= self.order - 1) + 0 + (min_zeros_value > 0) + 0) > 2
        if threshold.sum() > 0:
            min_zeros_value = min_zeros_value[threshold]
            self.mask_alpha[threshold, min_zeros_value] = 1
            qzx = feats.sum(1)
            qzx_norms = utils.norm(qzx, dim=-1)
            mus_new = qzx / qzx_norms

            kappas_new = 0.01 * torch.randn((self.num_models, self.order)).cuda()[threshold, min_zeros_value]
            alpha_new = 0.01 * torch.randn((self.num_models, self.order)).cuda()[threshold, min_zeros_value]
            self.alpha_logit[threshold, min_zeros_value] = nn.Parameter(alpha_new)
            self.mu_unnorm[threshold, min_zeros_value] = nn.Parameter(mus_new)[threshold]
            self.logkappa[threshold, min_zeros_value] = nn.Parameter(kappas_new)
            return 1
        else:
            return 0

    def get_params(self):
        alpha_logit = self.alpha_logit
        alpha_logit[self.mask_alpha == 0] = -999999
        logalpha = alpha_logit.log_softmax(1)
        mus = self.mu_unnorm / utils.norm(self.mu_unnorm, dim=-1)
        kappas = self.logkappa.exp() + self.reg

        return logalpha, mus, kappas, self.mask_alpha

    def forward(self, x, labels=None, original=None, original_feat=None, return_dot_xy=False, first_iter=False,
                after_init=True):

        '''
        Evaluate logliks, log p(x)

        Args:
            x = batch for x

        Returns:
            logliks = log p(x)
            logpcs = log p(x|c=m)

        '''

        with torch.no_grad():
            if first_iter:
                init_means = k_means_pp_init(x, self.order, num_models=self.num_models)
                self.mu_unnorm = copy.deepcopy(init_means.permute(1, 0, 2))
            logalpha, mu, kappa, masks = self.get_params()

            if (torch.isnan(self.logkappa).sum()) > 0:
                print("Here")
                mask_nan = torch.isnan(self.logkappa)
                self.logkappa[mask_nan] = nn.Parameter(0.01 * torch.randn(([]))).to(self.logkappa.device)
                kappa[mask_nan] = torch.zeros([]).to(kappa.device) + 0.1

            if return_dot_xy:
                dotxy = torch.matmul(return_norm(x[:, self.num_feat:].unsqueeze(0)),
                                     return_norm(mu[:, :, self.num_feat:]).permute(0, 2, 1))

            if first_iter:
                dotxy = torch.matmul(return_norm(x[:, self.num_feat:].unsqueeze(0)),
                                     return_norm(mu[:, :, self.num_feat:]).permute(0, 2, 1))
                dotp = dotxy
                logpcs = dotxy
                logpcs[(self.mask_alpha == 0).unsqueeze(1).repeat(1, logpcs.shape[1], 1)] = -999999
            else:
                dotp = torch.matmul(x.unsqueeze(0), mu.permute(0, 2, 1))
                logC = log_vmf_normalizer_approx(kappa ** 2, self.x_dim)
                logpcs = kappa.unsqueeze(1) * dotp + logC.unsqueeze(1)
            logliks_2 = (logalpha.unsqueeze(1) + logpcs)

            logliks = (logliks_2).logsumexp(2)
            if return_dot_xy:
                return logliks, logpcs, logliks_2, dotp, dotxy

            return logliks, logpcs, logliks_2, dotp

    def get_history(self,time=1):
        time_kernel = kernel_func(time,self.history_time)
        history_feat = (time_kernel.unsqueeze(-1).unsqueeze(-1)*self.history)[:,:].sum(1)
        history_q = (time_kernel.unsqueeze(-1)*self.history_q)[:,:].sum(1)


        return history_feat, history_q


    def update_time_ignore(self,time=1,add_time=0.33):
        for i in range(self.history_count):
            if self.history_time[:,i].max()>0:
                self.history_time[:,i] = self.history_time[:,i]+add_time



    def save_history(self, time=1, feat=None, q=None):
        num_of_original = 5
        if self.ind == 0:
            self.history[:, :num_of_original] = feat.unsqueeze(1)
            self.history_q[:, :num_of_original] = q.unsqueeze(1)
        ind_to_save = (self.ind % (self.history_count - num_of_original)) + num_of_original
        ind_to_save_1 = ((self.ind - 1) % (self.history_count - num_of_original)) + num_of_original
        ind_to_save_2 = ((self.ind - 1) % (self.history_count - num_of_original)) + num_of_original
        ind_to_save_3 = ((self.ind - 3) % (self.history_count - num_of_original)) + num_of_original

        ind_original_to_save_1 = ((self.ind - 1)) % num_of_original
        ind_original_to_save_2 = ((self.ind - 2)) % num_of_original

        self.history[:, self.ind % num_of_original, :, self.num_feat:] = feat[:, :, self.num_feat:] * (
                    (self.history_q[:, self.ind % num_of_original].unsqueeze(-1) + utils.realmin) / (
                        q.unsqueeze(-1) + utils.realmin))
        self.history_time[:, ind_to_save] = time
        self.history[:, ind_to_save] = feat
        self.history_q[:, ind_to_save] = q
        self.history_time[:, self.ind % num_of_original] = time - 0.33
        for i in range(num_of_original, self.history_count):
            if (i != ind_to_save_1 and i != ind_to_save and i != ind_to_save_2):
                self.history[:, i, :, self.num_feat:] = feat[:, :, self.num_feat:] * (
                            (self.history_q[:, i].unsqueeze(-1) + utils.realmin) / (q.unsqueeze(-1) + utils.realmin))

        for i in range(num_of_original):
            if (i != ind_original_to_save_1 and ind_original_to_save_2) and 0:
                self.history[:, i, :, self.num_feat:] = feat[:, :, self.num_feat:] * (
                            (self.history_q[:, i].unsqueeze(-1) + utils.realmin) / (q.unsqueeze(-1) + utils.realmin))

        self.ind = self.ind + 1

def ll_compute_batch(list_models, feat, args, mask_to_change, masks_to_keep, masks_to_ignore, scale=None,
                     gt_model_list=None, init_scale=1, clusters_masks=None, mask_to_change_tight=None,
                     list_of_ll_torch=None, buffer_for_clusters_masks=None, first_anno=None):
    scale = 1
    init_scale = 1
    with torch.no_grad():
        if first_anno is not None:
            first_anno = Image.fromarray(first_anno).resize((args.num_patches[1] * scale, args.num_patches[0] * scale),
                                                            Image.NEAREST)
            first_anno = torch.from_numpy(np.array(first_anno)).cuda().reshape(-1)
        if scale == init_scale:
            upsample_masks_to_keep = masks_to_keep
            upsample_mask_to_change = mask_to_change
            upsample_masks_to_ignore = masks_to_ignore

        if list_of_ll_torch is None:
            list_of_ll_torch = torch.zeros((args.num_items * args.num_models, feat.shape[0], args.model_order)).cuda()
        if buffer_for_clusters_masks is not None:
            new_clusters_masks = (buffer_for_clusters_masks[1] * 0).float()
            buffer_for_clusters_masks = buffer_for_clusters_masks[0] * 0
        else:
            new_clusters_masks = None

        list_of_ll = list_of_ll_torch * 0
        dotxy_list = []
        for i in range(len(list_models)):
            if args.ignore[i] == 1:
                dotxy_list.append(None)
                continue
            logliks, logpcs, logliks_2, dotp, dotxy = list_models[i](feat[upsample_mask_to_change], return_dot_xy=True)
            dotxy_list.append(dotxy)
            list_of_ll[(args.num_models * i):(args.num_models * (i + 1)), upsample_mask_to_change] = (logliks_2)
            list_of_ll[(args.num_models * i):(args.num_models * (i + 1)), upsample_masks_to_keep[i]] = 99999
            list_of_ll[(args.num_models * i):(args.num_models * (i + 1)), upsample_masks_to_ignore[i]] = 0

            if clusters_masks is not None and 1:
                if args.dotxy_value == None:
                    add_dotxy = -0.03

                    args.dotxy_value = torch.from_numpy(np.
                                                        linspace(0.68 + add_dotxy, 0.83 + add_dotxy,
                                                                 num=args.num_models)).cuda().unsqueeze(0).unsqueeze(0)
                    dotxy_value = args.dotxy_value * 1
                else:
                    dotxy_value = args.dotxy_value * 1

                if args.ignore[i] == 0:
                    list_of_ll[(args.num_models * i):(args.num_models * (i + 1)), upsample_mask_to_change] = list_of_ll[
                                                                                                             (
                                                                                                                     args.num_models * i):(
                                                                                                                     args.num_models * (
                                                                                                                     i + 1)),
                                                                                                             upsample_mask_to_change] * (
                                                                                                                     (
                                                                                                                             dotxy_list[
                                                                                                                                 i] > dotxy_value.permute(
                                                                                                                         2,
                                                                                                                         0,
                                                                                                                         1)) + 0).float()
                if buffer_for_clusters_masks is not None:
                    if args.ignore[i] == 0:
                        new_clusters_masks[:, upsample_mask_to_change,
                        args.model_order * i:args.model_order * (i + 1)] = (
                                (dotxy_list[i] > dotxy_value.permute(2, 0, 1)) + 0).float()
                        new_clusters_masks[:, upsample_masks_to_keep[i],
                        args.model_order * i:args.model_order * (i + 1)] = 1
                        new_clusters_masks[:, upsample_masks_to_ignore[i],
                        args.model_order * i:args.model_order * (i + 1)] = 0

        if first_anno is not None:
            for i in range(len(list_models)):
                list_of_ll[(args.num_models * i):(args.num_models * (i + 1)), ~(first_anno == i)] = 0

        list_of_ll[list_of_ll < 0] = 0.0

        prob_objects = (list_of_ll.max(-1)[0]).reshape(args.num_items, args.num_models, -1)
        prob_objects = prob_objects

        list_of_ll = list_of_ll.reshape(args.num_items, args.num_models, -1, args.model_order).permute(1, 2, 0,
                                                                                                       3).reshape(
            args.num_models, -1, args.model_order * args.num_items).argmax(-1)
        list_feat_to_return = []
        return prob_objects, 0, 0, list_of_ll.reshape(args.num_models, args.num_patches[0] * scale, args.num_patches[
            1] * scale), list_feat_to_return, list_of_ll_torch, new_clusters_masks


def log_vmf_normalizer_approx(k_squared, d):
  """Approximates log C_d(kappa) from the vMF probability density function.

  Args:
    k_squared: The value of the concentration parameter for a vMF distribution
      squared.
    d: Dimensionality of the embedding space.

  Returns:
    The approximation to log C_d(kappa).
  """
  d_m_half = (d / 2.0) - 0.5
  sqrt_d_m_half = torch.sqrt(d_m_half**2 + k_squared)

  d_p_half = (d / 2.0) + 0.5
  sqrt_d_p_half = torch.sqrt(d_p_half**2 + k_squared)

  return 0.5 * (
      d_m_half * torch.log(d_m_half + sqrt_d_m_half) - sqrt_d_m_half +
      d_m_half * torch.log(d_m_half + sqrt_d_p_half) - sqrt_d_p_half)


def run_em_batch(list_models, feat_list, args, save_history=None, get_history=None, original_feat=None, allow_new=True,
                 masks_to_change=None, use_knn=None):
    for i in range(args.num_items):
        if args.ignore[i] == 1:
            continue
        if args.dict_lastseen[str(i) + '_skip'] == 1:
            list_models[i].update_time_ignore()
            continue
        max_iters = args.max_iters
        if use_knn is not None and 1:
            # x_ind = torch.from_numpy(use_knn[1][i][:,0]*(args.num_patches[0]-1)).long()
            # y_ind = torch.from_numpy(use_knn[1][i][:,1]*(args.num_patches[1]-1)).long()
            x_ind = (use_knn[1][i][:, 0] * (args.num_patches[0] - 1)).long()
            y_ind = (use_knn[1][i][:, 1] * (args.num_patches[1] - 1)).long()
            test = use_knn[0]
            # test[:,~masks_to_change] = 1
            test = test.reshape(args.num_models, args.num_patches[0], args.num_patches[1],
                                args.model_order * args.num_items)
            test = test[:, x_ind, y_ind, i * args.model_order:(i + 1) * args.model_order]
            max_iters = 2
        if i == 0:
            treshold_for_new = 0.4
        else:
            treshold_for_new = 0.4

        # print("Item number: ",i)
        original_feat = None
        if original_feat is None:
            original_feat_current = None
        else:
            original_feat_current = original_feat[i].shape[0]
        create_new_cluster = 1
        start_over = 0
        # original_feat_current=None

        # feat = (np.concatenate((feat_list[i],dist_list[i]),-1))

        feat = feat_list[i]

        ll_old = -np.inf
        with torch.no_grad():
            if get_history is not None:
                history_feat, history_q = list_models[i].get_history(time=get_history)
                history_feat = history_feat
                history_q = history_q
                sum_history_q = history_q[0].sum()
            else:
                history_feat = 0
                history_q = 0
                sum_history_q = 0

            for steps in range(max_iters):
                if get_history is None and steps == 0:
                    first_iter = True
                else:
                    first_iter = False
                logalpha, mus, kappas, masks = list_models[i].get_params()
                logliks, logpcs, jll, dotp = list_models[i](feat, original_feat=original_feat_current,
                                                            first_iter=first_iter)
                ll = logliks.sum(-1).max()
                if use_knn is not None and 1:
                    jll = jll * test
                qz = jll.log_softmax(2).exp()

                # tolerance check_
                if steps > (3 + 10 * start_over):
                    if allow_new and 0:
                        threshold_new = dotp.max(-1)[0].min(-1)[0] < treshold_for_new
                        thershold_points = dotp.max(-1)[0].argmin(-1)
                        sum_of = (dotp.max(-1)[0] < treshold_for_new).sum(-1).float() / feat.shape[0] > 0.02
                        if feat.shape[0] > 50:
                            knn = 50
                        else:
                            knn = feat.shape[0]
                        topk_val, topk_ind = torch.topk(
                            torch.matmul(feat[thershold_points].unsqueeze(0), feat.T.unsqueeze(0)), k=knn, dim=-1,
                            sorted=False)
                        threshold_new = (threshold_new + 0 + sum_of + 0) > 1
                        if ((threshold_new + 0).sum() > 0) and start_over == 0 and allow_new and 1:
                            create_new_cluster = list_models[i].open_new_cluster(threshold_new, feat[topk_ind[0]])
                            if create_new_cluster:
                                start_over = start_over + 1
                                continue

                    rll = (ll - ll_old).abs() / (ll_old.abs() + utils.realmin)
                    if rll < args.rll_tol:
                        break

                ll_old = ll
                # M-step

                feat_to_send = feat
                qzx_prev = torch.einsum("abcd,abcd -> acd", qz.unsqueeze(3), feat_to_send.unsqueeze(0).unsqueeze(2))
                if get_history is not None:
                    # history_feat[:,:,384:] = qzx_prev[:,:,384:]/(qz.sum(1)[0].sum()/sum_history_q)
                    a = 0
                qzx = qzx_prev + history_feat
                qzx_norms = utils.norm(qzx, dim=-1)
                mus_new = qzx / qzx_norms
                Rs = (qzx_norms[:, :, -1]) / (
                            qz.sum(1) + history_q + utils.realmin * qzx_norms.shape[0])  # .clamp(0.01,0.99)
                ### for numircal stability we should clamp
                kappas_new = ((list_models[i].x_dim * Rs - Rs ** 3) / ((1 - Rs ** 2))).clamp(utils.realmin, 9000)
                alpha_new = (qz.sum(1) + history_q + utils.realmin) / (
                            qz.sum(1)[0].sum() + sum_history_q + utils.realmin)

                if (torch.isinf(kappas_new).sum() > 0):
                    mask = torch.isinf(kappas_new)
                    # kappas_new[mask] = torch.nn.Parameter(torch.abs(torch.randn(([])))).cuda()
                    kappas_new[mask] = copy.deepcopy(kappas_new[~mask][0])
                    mus_new[mask] = copy.deepcopy(mus_new[~mask][0].unsqueeze(0))
                    # mus_new[mask] = torch.nn.Parameter(torch.randn((mus_new[0,0]).shape)).cuda()
                    # list_models[i].set_mask(masks*((mask==0)+0))
                    print("I")
                if (torch.isnan(kappas_new).sum() > 0):
                    mask = torch.isnan(kappas_new)
                    # kappas_new[mask] = torch.nn.Parameter(torch.abs(torch.randn(([])))).cuda()
                    # mus_new[mask] = torch.nn.Parameter(torch.randn((mus_new[0,0]).shape)).cuda()
                    kappas_new[mask] = copy.deepcopy(kappas_new[~mask][0])
                    mus_new[mask] = copy.deepcopy(mus_new[~mask][0].unsqueeze(0))
                    print("N")
                if ((kappas_new <= 0).sum() > 0):
                    mask = kappas_new <= 0
                    # kappas_new[mask] = torch.nn.Parameter(torch.abs(torch.randn(([])))).cuda()
                    # mus_new[mask] = torch.nn.Parameter(torch.randn((mus_new[0,0]).shape)).cuda()
                    kappas_new[mask] = copy.deepcopy(kappas_new[~mask][0])
                    mus_new[mask] = copy.deepcopy(mus_new[~mask][0].unsqueeze(0))
                    print("Z")

                # assign new params
                list_models[i].set_params(alpha_new, mus_new, kappas_new)
        if save_history is not None:
            # qzx_prev[:,:,-5:]=0
            list_models[i].save_history(save_history, qzx_prev, qz.sum(1))
            # list_models[i].save_history(save_history,qzx,qz.sum(1)+history_q)

def crf_seg(crf_net, args, first_anno, img_lab_resize_torch, ind, pred_crf_1):
    image = img_lab_resize_torch.unsqueeze(0).permute(0, 3, 1, 2).float()
    unary = pred_crf_1.unsqueeze(0)

    if ind == 0:
        range_to_send = 100
        lr = 0.02
        print("Optimizing CRF")
        crf_net.train()
        optimizer = torch.optim.Adam(crf_net.parameters(), lr=lr,amsgrad=True)

        label = torch.nn.functional.one_hot(first_anno.unsqueeze(0).long(),
                                            num_classes=args.num_items).permute(0, 3, 1, 2).float()
        loss_fn = Dual_Focal_loss().cuda()
        for epoch in range(range_to_send):

            image_to_send = (image+10.0*torch.randn(image.shape).cuda()).clip(0,255)
            unary_to_send = unary

            crf_net.train()
            pred = crf_net(unary_to_send,image_to_send ).mean(0).unsqueeze(0)
            loss = loss_fn(pred, label.argmax(1))
            optimizer.zero_grad()
            loss = loss
            loss.backward()
            optimizer.step()
        crf_net.eval()
    with torch.no_grad():
        pred = crf_net(unary, image, create_position=ind == 0, use_preposition=ind > 0)
    return pred

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
        mask_hard_test = copy.deepcopy(mask_hard_distance)*0
        if distance_normal is not None and 1:
            current_distance_normal = distance_normal
            mask_hard_distance =(((labels_current==val)+0) +(current_distance_normal))>1
            if masks_to_change is not None:
                mask_hard_distance = (mask_hard_distance+0+masks_to_change.reshape(-1)+0)>1
            mask_hard_test = (((labels_current==val)+0)+ ((~mask_hard_distance))+0)>1



        current_feat_to_send_soft = feat_norm[mask_hard_distance]
        if xy_origin_to_use is None:
            xy_origin = xy[mask_hard_distance]
        else:
            xy_origin = xy_origin_to_use[mask_hard_distance]
        feat_norm_list.append(current_feat_to_send_soft)
        dist_list.append(xy_origin/args.xy_scale)

    return feat_norm_list,test_for_outliers,dist_list

class Dual_Focal_loss(nn.Module):
    '''
    This loss is proposed in this paper: https://arxiv.org/abs/1909.11932
    '''

    def __init__(self, ignore_lb=255, eps=1e-5, reduction='mean'):
        super(Dual_Focal_loss, self).__init__()
        self.ignore_lb = ignore_lb
        self.eps = eps
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, logits, label):
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1).detach()

        pred = torch.softmax(logits, dim=1)
        loss = -torch.log(self.eps + 1. - self.mse(pred, lb_one_hot)).sum(dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss
