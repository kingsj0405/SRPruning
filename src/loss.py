import torch
import torch.nn.functional as F


def PatchMSELoss(output, target):
    new_H, h_self, h_others = localPatchDist(output, target.unsqueeze(2), 7, 9)
    loss_Pixel = torch.mean((output - new_H) ** 2)
    return loss_Pixel

    
def localPatchDist(batch_S, batch_H, k_size=5, s_size=9):
    ##
    # batch_S : [B, C, H, W]
    # batch_H : [B, C, T, H, W]

    # if k_size == 3:
    #     ret_layer = 2
    # elif k_size == 5:
    #     ret_layer = 3

    # TOP_K = 5

    #
    ds = s_size - k_size + 1

    p_size_k = k_size // 2
    B, C, T, H, W = batch_H.size()
    # batch_S = F.pad(batch_S, [p_size_k,p_size_k,p_size_k,p_size_k], mode='reflect')
    # _, _, pH, pW = batch_S.size()
    unfold_S = F.unfold(batch_S, k_size, dilation=1, padding=0, stride=k_size)   # B, C*k_size*k_size, L
    _, _, L = unfold_S.size()
    unfold_S = unfold_S.permute(0, 2, 1).reshape(-1, C, k_size, k_size)   # B*L, C, k_size, k_size
    # print(unfold_S.size())
    # batch_F_S = model_F(unfold_S, ret=ret_layer)  # B*L, C', 1, 1


    with torch.no_grad():
        # batch_H_target = F.pad(batch_H[:,:,T//2], [p_size_k,p_size_k,p_size_k,p_size_k], mode='reflect')
        # unfold_H_target = F.unfold(batch_H_target, k_size, dilation=1, padding=0, stride=1)   # B, C*k_size*k_size, L
        # unfold_H_target = unfold_H_target.permute(0, 2, 1).reshape(-1, C, k_size, k_size)   # B*L, C, k_size, k_size
        # batch_F_H_target = model_F(unfold_H_target, ret=ret_layer)  # B*L, C', 1, 1


        # p_size_s = s_size // 2 
        unfolds = []

        for t in range(T//2, T//2+1):
            batch_Ht = F.pad(batch_H[:,:,t], [ds//2,ds//2,ds//2,ds//2], mode='reflect')
            for y in range(ds):
                for x in range(ds):
                    unfolds += [ F.unfold(batch_Ht[:, :, y:y+H, x:x+W], k_size, dilation=1, padding=0, stride=k_size) ]
                            
        unfold_H = torch.stack(unfolds, 3)    # B, C*k_size*k_size, L, T*ds*ds
        unfold_H = unfold_H.permute(0, 2, 1, 3).reshape(B*L, C, k_size, k_size, -1)   # B*L, C, k_size, k_size, T*ds*ds
        # print(unfold_H.size())
        # batch_F_H = model_F(unfold_H, ret=ret_layer)  # B*L*T*ds*ds, C', 1, 1
        # batch_F_H = batch_F_H.reshape(B*L, T*ds*ds, -1, 1, 1).permute(0, 2, 3, 4, 1)  # B*L, C', 1, 1, T*ds*ds

        # Gen-GT matching
        loss_patch = torch.mean((unfold_S.unsqueeze(4) - unfold_H)**2, dim=[1,2,3]) # B*L, T*ds*ds
        patch_min = torch.min(loss_patch, dim=1)    # B*L

        patch_min_ind = patch_min[1] + torch.arange(0, B*L*1*ds*ds, 1*ds*ds).cuda()    # B*L

        # Top similar patches
        unfold_H = unfold_H.permute(0, 4, 1, 2, 3).reshape(B*L*1*ds*ds, C, k_size, k_size)   # B*L*T*ds*ds, C, k_size, k_size
        unfold_H_selected = torch.index_select(unfold_H, 0, patch_min_ind).detach()   # B*L, C, k_size, k_size

    # unfold_S = F.unfold(batch_S, k_size, dilation=1, padding=0, stride=k_size)   # B, C*k_size*k_size, L
    unfold_H_selected = unfold_H_selected.reshape(B, L, -1).permute(0, 2, 1)
    # print(unfold_H_selected.size())
    fold_H_selected = F.fold(unfold_H_selected, [H, W], k_size, stride=k_size)    #[:, :, p_size_k:-p_size_k, p_size_k:-p_size_k]
    # print(fold_H_selected.size())
    

    # input(fold_H_selected.size())

    # # Calc loss
    # mse_margin = ((1/255.0)/2.0)**2
    # loss_pixel_patch = torch.nn.ReLU()((unfold_S - unfold_H_selected)**2 - mse_margin).mean()



    # batch_F_S = model_F(unfold_S)
    # batch_F_H = model_F(unfold_H_selected)

    # loss_VGG_patches = []
    # for f in [0,1,2,3,4]:
    #     loss_VGG_patches += [ torch.mean((batch_F_S[f] - batch_F_H[f])**2) ]
    # loss_VGG_patch = torch.mean(torch.stack(loss_VGG_patches))


    # hit
    hit_self = torch.sum(torch.eq(patch_min[1], torch.zeros_like(patch_min[1])+(ds*ds)//2)).item()
    hit_others = torch.sum(torch.ne(patch_min[1], torch.zeros_like(patch_min[1])+(ds*ds)//2)).item()


    return fold_H_selected.detach(), hit_self, hit_others


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        return loss 


loss_map = {
    'PatchMSELoss': PatchMSELoss,
    'MSELoss': torch.nn.MSELoss(),
    'CharbonnierLoss': L1_Charbonnier_loss(),
}


def get_loss(loss_name):
    if loss_name not in loss_map.keys():
        raise ValueError(f"{loss_name} is not in loss_map")
    else:
        return loss_map[loss_name]