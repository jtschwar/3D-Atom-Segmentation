import models.utils as utils
import torch
import torch.nn.functional as F
X = 2
Y = 4
Z = A = 8
B = torch.randn((X, Y, Z, A))

up = utils.InterpolateUpsampling()

tt = F.interpolate(B,size=[Z*3,A*3])

# tt = up(B,[Z*3, A*3])

# b_usample = Fun.interpolate(B, [Z*3, A*3], mode='bilinear', align_corners=True)
# b_mod = B.clone()
# b_mod[:, 0] *= 2000
# b_mod_usample = Fun.interpolate(b_mod, [Z*3, A*3], mode='bilinear', align_corners=True)
# print(torch.isclose(b_usample[:,0], b_mod_usample[:,0]).all())
# print(torch.isclose(b_usample[:,1], b_mod_usample[:,1]).all())
# print(torch.isclose(b_usample[:,2], b_mod_usample[:,2]).all())
# print(torch.isclose(b_usample[:,3], b_mod_usample[:,3]).all())