import torch
import torch.nn as nn
import torch.autograd as grad
import math

class DenseTriangularFlow(nn.Module):
    """
        A dense neural layer, restricted to triangular weights for invertability, and tractable log determinant
    """
    def __init__(self, dim, upper):
        super(DenseTriangularFlow, self).__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.upper = upper

    def w_tri(self):
        if self.upper:
            return self.w.triu()
        else:
            return self.w.tril()
        
    def forward(self, x):
        x = x.unsqueeze(2)
        x = torch.matmul(self.w_tri(), x)
        x = x.squeeze(2)
        x = x + self.b
        log_det = self.w.diag().log().sum()
        return x, log_det

    def backward(self, y):
        y = y - self.b
        y = y.unsqueeze(2)
        x, _ = torch.triangular_solve(y, self.w_tri(), upper=self.upper)
        x = x.squeeze(2)
        log_det = -1.0 * self.w.diag().log().sum()
        return x, log_det

# TODO: implement a sigmoid flow that mixes in a bit of x, to be invertible for the full domain of y
class SigmoidFlow(nn.Module):
    """
        A sigmoid activation function, with inverse and log determinant
    """
    def __init__(self):
        super(SigmoidFlow, self).__init__()
        self.sig = nn.Sigmoid()

    def dydx(self, x):
        e_pow_negx = (-1.0 * x).exp()
        return e_pow_negx / (e_pow_negx + 1) ** 2

    def forward(self, x):
        log_det = self.dydx(x).log().sum(1)
        return self.sig(x), log_det

    def backward(self, y):
        x = (-y / (y-1)).log()
        log_det = - self.dydx(x).log().sum(1)
        return x, log_det

class SoftlogFlowFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign() * (input.abs() + 1).log()

    def dydx(x):
        grad = x.abs() / (x.abs() + x * x)
        grad[x == 0.] = 1.
        return grad

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        dydx = SoftlogFlowFunction.dydx(input)
        return grad_output * dydx

# Hmm, could do a similar thing with y = exp(abs(x)) - 1
class SoftlogFlow(nn.Module):
    """
        A non-linear activation function, with analytic inverse and log determinant.

        This activation function has the domain and range of R, and thus can be used to build invertible networks.

        Equation: y = sign(x) log(abs(x) + 1)
    """
    def __init__(self):
        super(SoftlogFlow, self).__init__()

    def forward(self, x):
        softlog = SoftlogFlowFunction.apply
        log_det = SoftlogFlowFunction.dydx(x).log().sum(1)
        return softlog(x), log_det

    def backward(self, y):
        x = y.sign() * (y.abs().exp() - 1)
        log_det = - SoftlogFlowFunction.dydx(x).log().sum(1)
        return x, log_det

class SoftsquareFlowFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, x):
        ctx.save_for_backward(a, b, x)
        return a * x + b * x.sign() * x * x

    def dydx(a, b, x):
        return a + 2.0 * b * x.sign() * x

    @staticmethod
    def backward(ctx, grad_output):
        a, b, x = ctx.saved_tensors
        dydx = SoftsquareFlowFunction.dydx(a, b, x)
        dyda = x
        dydb = x.sign() * x * x
        return dyda * grad_output, dydb * grad_output, dydx * grad_output

class SoftsquareFlow(nn.Module):
    """
        A non-linear, residual activation function, with analytic inverse and log determinant.

        This activation function has the domain and range of R, and thus can be used to build invertible networks.

        Equation: y = a * x + b * sign(x) * x * x
    """
    def __init__(self, dim):
        super(SoftsquareFlow, self).__init__()
        self.a = nn.Parameter(torch.ones(dim))
        # the gradient of 0.abs() seems to evaluate to zero.
        # we need to add a small epsilon so b can be trained
        self.b = nn.Parameter(torch.zeros(dim) + 1e-16)

    def forward(self, x):
        a = self.a.abs()
        b = self.b.abs()

        softsquare = SoftsquareFlowFunction.apply

        log_det = SoftsquareFlowFunction.dydx(a, b, x).log().sum(1)
        return softsquare(a, b, x), log_det

    def backward(self, y):
        a = self.a.abs()
        b = self.b.abs()

        if b != 0.0:
            aa = a * a
            by4 = 4.0 * b * y

            root = (aa + by4.abs()).sqrt()
            x = y.sign() * (root - a) / (2.0 * b)
        else:
            x = y / a

        log_det = - SoftsquareFlowFunction.dydx(a, b, x).log().sum(1)
        return x, log_det

class InverseFlow(nn.Module):
    """
        An inverse sigmoid activation function.  Reverses the flow implementation.
    """
    def __init__(self, flow):
        super(InverseFlow, self).__init__()
        self.flow = flow
    
    def forward(self, x):
        y, log_det = self.flow.backward(x)
        return y, log_det

    def backward(self, y):
        x, log_det = self.flow.forward(y)
        return x, log_det

class RankOneConvolutionFlow(nn.Module):
    """
        Multiplies the input by I + vT u, a rank-one update from the identity matrix.

        This transformation has a tractable log determinant, and an easily calculable inverse.
    """

    def __init__(self, dim, epsilon=1e-16):
        super(RankOneConvolutionFlow, self).__init__()
        self.u = nn.Parameter(torch.zeros(dim, 1) + epsilon)
        self.vT = nn.Parameter(torch.zeros(1, dim) + epsilon)
    
    def forward(self, x):
        vT_x = torch.matmul(self.vT, x.unsqueeze(2))
        y = x + torch.matmul(self.u, vT_x).squeeze(2)
        det = 1. + torch.matmul(self.vT, self.u).squeeze()
        return y, det.log()

    def backward(self, y):
        vT_y = torch.matmul(self.vT, y.unsqueeze(2))
        vT_u = torch.matmul(self.vT, self.u)
        u_vT_y = torch.matmul(self.u, vT_y).squeeze(2)
        x = y - 1. / (1 + vT_u) * u_vT_y
        det = 1. / (1 + vT_u)
        return x, det.log()

# TODO: make the Cdf and Loss generic over a pytorch distribution
# Maybe there is a way to compute CdfFlow.dydx with autodiff?
class NormalCdfFlow(nn.Module):
    """
        Converts a uniformly-distributed variable to a normally distributed variable using the Inverse CDF.
    """
    def __init__(self):
        super(NormalCdfFlow, self).__init__()
        self.norm = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    
    def dydx(self, x):
        return torch.exp(-(x*x)/2.0) / math.sqrt(2. * math.pi)

    def forward(self, x):
        y = self.norm.cdf(x)
        log_det = self.dydx(x).log().sum(1)
        return y, log_det

    def backward(self, y):
        x = self.norm.icdf(y)
        log_det = -1.0 * self.dydx(x).log().sum(1)
        return x, log_det

class NegLogLikelihoodLoss(nn.Module):
    """
        Negative log likelihood loss, assuming the data is from a multivariate gaussian with an identity covariance matrix.

        Includes a penalty term for the log determinant.  This prevents 'squeezing' down to the space where the gaussian PDF is at it's maximum.
    """
    def __init__(self, dim):
        super(NegLogLikelihoodLoss, self).__init__()
        self.dim = dim

    def forward(self, x, log_det):
        k = self.dim
        xt_x = (x * x).sum(-1)
        log_pdf = math.log(pow(2 * math.pi, -k / 2)) - xt_x / 2
        objective = log_pdf + log_det
        return -1.0 * objective.mean()

class Flows(nn.Module):
    """
        Composes flows together.  Sequences inputs and outputs, sums log determinants, and implements forwards/reverse
    """

    def __init__(self, *args):
        super(Flows, self).__init__()
        self.flows = nn.ModuleList(args)

    def forward(self, x):
        v = x
        log_det = torch.tensor(0.0)
        for mod in iter(self.flows):
            v, new_log_det = mod(v)
            log_det = log_det + new_log_det

        return v, log_det

    def forward_trace(self, x):
        v = x
        log_det = torch.tensor(0.0)

        vs = []
        dets = []

        for mod in self.flows:
            v, new_log_det = mod.forward(v)
            vs.append(v)
            log_det = log_det + new_log_det
            dets.append(log_det)
        
        return vs, dets

    def backward(self, y):
        v = y
        log_det = torch.tensor(0.0)

        for mod in reversed(self.flows):
            v, new_log_det = mod.backward(v)
            log_det = log_det + new_log_det

        return v, log_det

    def backward_trace(self, y):
        v = y
        log_det = torch.tensor(0.0)

        vs = []
        dets = []

        for mod in reversed(self.flows):
            v, new_log_det = mod.backward(v)
            vs.append(v)
            log_det = log_det + new_log_det
            dets.append(log_det)

        return vs, dets

class FlowModule(nn.Module):
    """
        Composes a Flow with an loss function
    """
    def __init__(self, flow, loss):
        super(FlowModule, self).__init__()
        self.flow = flow
        self.loss = loss

    def forward(self, x):
        y, log_det = self.flow(x)
        return self.loss(y, log_det)