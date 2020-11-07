class HeatmapGenerator:
    def __init__(self, module):
        self.backward_hook = module.register_backward_hook(self._backward_hook_func)
        self.forward_hook = module.register_forward_hook(self._forward_hook_func)
    
    def _backward_hook_func(self, m, gi, go):
        self.grad = go[0].detach().clone()
    
    def _forward_hook_func(self, m, i, o):
        self.act = o.detach().clone()
    
    def __enter__(self, *args):
        return self
    
    def __exit__(self, *args):
        self.backward_hook.remove()
        self.forward_hook.remove()
    
    def get_heatmaps(self):
        w = self.grad.mean(dim=[2,3], keepdim=True)
        return (w * self.act).sum(1)