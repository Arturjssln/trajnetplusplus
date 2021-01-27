"""
Code imported from https://github.com/visinf/mar-scf
"""
from __future__ import print_function
from collections import OrderedDict
import math
import torch
import torch.nn as nn

def arccosh(x):
    return torch.log(x + torch.sqrt(x.pow(2)-1))

def arcsinh(x):
    return torch.log(x + torch.sqrt(x.pow(2)+1))


class Split2dMsC(nn.Module):
	def __init__(self, level=0):
		super().__init__()
		self.level = level

	def split_feature(self, z):
		return z[:,:z.size(1)//2], z[:,z.size(1)//2:]

	def forward(self, input, logdet=0., reverse=False, nn_outp=None):
		if not reverse:
			z1, z2 = self.split_feature(input)
			return ( z1, z2), logdet
		else:
			z1, z2 = input
			z = torch.cat((z1, z2), dim=1)
			return z, logdet					

class NLFlowStep(torch.nn.Module):
    """ 
    Sub-network of the Normalizing flow network

    """
    def __init__(self):
        super(NLFlowStep, self).__init__()
        self.num_params = 5
        self.logA = math.log(8*math.sqrt(3)/9-0.05) # 0.05 is a small number to prevent exactly 0 slope


    def get_pseudo_params(self, nn_outp):
        a = nn_outp[..., 0] # [B, D]
        logb = nn_outp[..., 1]*0.4
        B = nn_outp[..., 2]*0.3
        logd = nn_outp[..., 3]*0.4
        f = nn_outp[..., 4]

        b = torch.exp(logb)
        d = torch.exp(logd)
        c = torch.tanh(B)*torch.exp(self.logA + logb - logd)

        return a, b, c, d, f


    def forward_inference(self, x, nn_outp):
        a, b, c, d, f = self.get_pseudo_params(nn_outp)
        
        # double needed for stability. No effect on overall speed
        a = a.double()
        b = b.double()
        c = c.double()
        d = d.double()
        f = f.double()
        x = x.double()

        aa = -b*d.pow(2)
        bb = (x-a)*d.pow(2) - 2*b*d*f
        cc = (x-a)*2*d*f - b*(1+f.pow(2))
        dd = (x-a)*(1+f.pow(2)) - c

        p = (3*aa*cc - bb.pow(2))/(3*aa.pow(2))
        q = (2*bb.pow(3) - 9*aa*bb*cc + 27*aa.pow(2)*dd)/(27*aa.pow(3))
        
        t = -2*torch.abs(q)/q*torch.sqrt(torch.abs(p)/3)
        inter_term1 = -3*torch.abs(q)/(2*p)*torch.sqrt(3/torch.abs(p))
        inter_term2 = 1/3*arccosh(torch.abs(inter_term1-1)+1)
        t = t*torch.cosh(inter_term2)

        tpos = -2*torch.sqrt(torch.abs(p)/3)
        inter_term1 = 3*q/(2*p)*torch.sqrt(3/torch.abs(p))
        inter_term2 = 1/3*arcsinh(inter_term1)
        tpos = tpos*torch.sinh(inter_term2)

        t[p > 0] = tpos[p > 0]
        y = t - bb/(3*aa)

        arg = d*y + f
        denom = 1 + arg.pow(2)

        x_new = a + b*y + c/denom

        logdet = -torch.log(b - 2*c*d*arg/denom.pow(2)).sum(0)

        y = y.float()
        logdet = logdet.float()

        return y, logdet

    def reverse_sampling(self, y, nn_outp):
        a, b, c, d, f = self.get_pseudo_params(nn_outp)

        arg = d*y + f
        denom = 1 + arg.pow(2)
        x = a + b*y + c/denom

        logdet = -torch.log(b - 2*c*d*arg/denom.pow(2)).sum(0)

        return x, logdet

    def forward(self, input, nn_outp, reverse):
        if not reverse:
            z, logdet = self.forward_inference(input, nn_outp)
        else:
            z, logdet = self.reverse_sampling(input, nn_outp)
        return z, logdet



class FlowNet(nn.Module):
    """ 
    Normalizing flow network
    """
    def __init__(self, nb_layers, latent_size):
        super(FlowNet, self).__init__()
        self.layers = nn.ModuleList()
        self.nb_layers = nb_layers
        self.half_latent_size = latent_size//2

        # FlowSteps
        for i in range(self.nb_layers):
            self.layers.append(NLFlowStep())
                

    def forward(self, input, logdet=0., reverse=False):
        """
        Parameters
        ----------
        input : Tuple(Tensor [batch_size, latent_size], Tensor [batch_size, nb_layers, 5])
            First value of the tuple: 
                Input value of the normalizing flow, sampled value from the latent space
                If reverse is True, it should the inputs to normalize
                If reverse is Flase, it should the normalized inputs from which samples will be generated
            Second value of the tuple: 
                Coefficients of the Normalizing flow network
        logdet : Double
            Will contain the logarithm of the determinant (used for loss)
            Default: 0.
        reverse : Boolean
            Define if the Normalizing flow should be used in normal or reverse mode
            Default: False

        Return
        -------
        loss : Tuple(Tensor [batch_size, latent_size], Tensor [batch_size, 1])
            Return a tuple of the output value and the logarithm of the determinant of jacobians
            If reverse is True, it will return the normalized values (as first element)
            If reverse is Flase, it will return the generated values (as first element)
        """
        if not reverse:
            return self.encode(input, logdet=logdet)
        else:
            return self.decode(input, logdet=logdet)

    def encode(self, input, logdet=0.0):
        z, coefs = input
        for i, layer in enumerate(self.layers):
            z, delta_logdet = layer(z, reverse=False, nn_outp=coefs[:, i, :])
            logdet += delta_logdet
        return z, logdet

    def decode(self, input, logdet=0.0):
        z, coefs = input
        for i, layer in reversed(list(enumerate(self.layers))):
            z, delta_logdet = layer(z, reverse=True, nn_outp=coefs[:, i, :])
            logdet += delta_logdet
        return z, logdet
