import random
import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['OmniContrastiveFeatureLoss']


class OmniContrastiveFeatureLoss(nn.Module):
    def __init__(self, 
                student_channels=512, 
                teacher_channels=2048, 
                tau_ocd=0.07, 
                M_ocd=16, 
                pool_size=4,
                patch_size=(4,4),
                rand_mask=True, 
                mask_ratio=0.75,
                enhance_projector=False,
                dataset='citys'
                ):

        super(OmniContrastiveFeatureLoss, self).__init__()

        self.zeta_fd = mask_ratio
        self.tau_ocd = tau_ocd
        self.M_ocd = M_ocd
        self.pool_size = pool_size
        self.patch_size = patch_size
        self.dataset = dataset
        self.rand_mask = rand_mask

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)

        self.generator = None
        if enhance_projector:
            self.projetor = EnhancedProjector(teacher_channels, teacher_channels)
        else:
            self.projetor = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))


    def forward(self, feat_S, feat_T):
        feat_S = self.align(feat_S)
        dis_loss, ctr_loss = self.get_dis_loss(feat_S, feat_T)
        
        return dis_loss, ctr_loss

    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        BS, C, H, W = preds_T.shape

        device = preds_S.device

        if self.rand_mask:
            mat = torch.rand((BS,1,H,W)).to(device)
            mat = torch.where(mat > 1-self.zeta_fd, 0, 1).to(device)
        else:
            mat = torch.ones((BS,1,H,W)).to(device)
            mat[:,:,H//3:2*H//3, W//3:2*W//3] = 0

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.projetor(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/BS
        ctr_loss = self.get_omni_contrastive_loss(new_fea, preds_T)/BS
    
        return dis_loss, ctr_loss

    def get_omni_contrastive_loss(self, preds, targets):
        loss_ce = nn.CrossEntropyLoss(reduction='sum')
        device = preds.device
        if self.dataset == 'camvid':
            # Camvid
            preds   = preds[:,:,:44,:44]
            targets = targets[:,:,:44,:44]
        else:
            # City and ADE20k and Pascal VOC
            if self.pool_size != 0:
                preds   = F.max_pool2d(preds, self.pool_size)
                targets = F.max_pool2d(targets, self.pool_size)
    
        BS, C, H, W = preds.shape
        N2 = self.patch_size[0] * self.patch_size[1]
        M = self.M_ocd

        preds   = self._to_ctr_format(preds, patch_size=self.patch_size)
        targets = self._to_ctr_format(targets, patch_size=self.patch_size)

        similarity_matrix = - torch.cdist(preds, targets, p=1)

        mask = torch.eye(N2*M, dtype=torch.bool).to(device)
        mask = mask.repeat(similarity_matrix.shape[0],1,1)

        mask_c = torch.cat([torch.arange(M) for i in range(N2)], dim=0).to(device)
        mask_c = (mask_c.unsqueeze(0) == mask_c.unsqueeze(1)).bool()
        mask_negative = ~ mask_c.repeat(similarity_matrix.shape[0],1,1)

        positives = similarity_matrix[mask].view(BS*H*W*M, -1)
        negatives = similarity_matrix[mask_negative].view(BS*H*W*M, -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros((logits.shape[0]), dtype=torch.long).to(device)

        logits = logits/self.tau_ocd
        
        ctr_loss = loss_ce(logits, labels)
        ctr_loss /= (H*W)

        return ctr_loss

    def _to_ctr_format(self, x, patch_size=(4,4)):
        M = self.M_ocd
        BS, C, H, W = x.shape
        H_p, W_p = patch_size
        num_p = (H*W)//(H_p*W_p)
        N2 = H_p * W_p

        x = x.contiguous().view(BS, C, H//H_p, H_p, W//W_p, W_p).permute(0,2,4,1,3,5)
        x = torch.stack(x.split(C//M, dim=3), dim=3)
        x = x.contiguous().view(BS, num_p, M, C//M, N2)
        x = x.permute(0,1,4,2,3).contiguous().view(-1, N2 * M, C//M)

        return x

class EnhancedProjector(nn.Module):
    def __init__(self, 
                in_channels=2048, 
                out_channels=2048, 
                ):
        super(EnhancedProjector, self).__init__()
        self.block_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True))
        self.block_2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True))
        self.adpator_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adpator_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)

        x_1 = self.adpator_1(x_1)
        x_2 = self.adpator_2(x_2)

        out = (x_1 + x_2)/2.

        return out

    






