import torch
import torchvision


def HookedResnet(object):
    def __init__(self,
                 arch,
                 n_classes,
                 device,
                 layer_names):
        self.model = torchvision.models.__dict__[arch](num_classes=n_classes)
        self.device = device
        self.layer_names = layer_names
        
        self.activations = {}
        self.hooks = {}
        
        
    def restore_weights(self,
                        checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # Get rid of 'module' from keys (from multi-GPU training)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        
    def get_activation(self, layer_name):
        def hook(model, input, output):
            self.activations[layer_name].append(output.detach().cpu().numpy())
        
        return hook
    def _register_hooks(self):
        # Reset activations dictionary
        for layer_name in self.layer_names:
            self.activations[layer_name] = []
            hook_code = "self.model.{}.register_forward_hook(self.get_activation('{}')".format(layer_name, layer_name)
            self.hooks[layer_name] = exec(hook_code)
            
    def _remove_hooks(self):
        for layer_name, hook in self.hooks.items():
            hook.remove()
            
    def run_examples(self, data):
        self._register_hooks()
        
        self.model.eval()
        self.model = self.model.to(device)
        
        data = torch.tensor(data)
        if len(data.shape) == 4 and data.shape[-1] == 3:
            data = data.permute(data, (0, 3, 1, 2))
        data = data.to(device)
        
        model(data)
        
        self._remove_hooks()
        
        return self.activations 
            
        
        
        
    