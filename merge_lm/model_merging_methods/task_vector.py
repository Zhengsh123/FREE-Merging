import torch
import torch.nn as nn

from utils_file.utils import get_param_names_to_merge


class TaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]

    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        # assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]

        return merged_params
    

class TaskVectorFFT:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None,high_ratio_for_fourier:float=0.6):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        self.high_ratio_for_fourier=high_ratio_for_fourier
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]
            self.find_low_high_fft(param_names_to_merge)

    def find_low_high_fft(self,param_names_to_merge):
        for key in self.task_vector_param_dict.keys():
            if key in param_names_to_merge and 'weight' in key and self.task_vector_param_dict[key].ndim==2:
                low_freq_spatial,high_freq_spatial,band_pass_spatial=self.cal_low_and_high_fft_for_tensor(self.task_vector_param_dict[key])
                self.task_vector_param_dict[key]=band_pass_spatial+low_freq_spatial#+high_freq_spatial
            # elif 'weight' in key and 'ln' in key:
            #     low_freq_spatial,high_freq_spatial,band_pass_spatial=self.cal_low_and_high_fft_for_tensor_1D(self.vector[key])
            #     self.vector[key]=band_pass_spatial+low_freq_spatial+high_freq_spatial

    def cal_low_and_high_fft_for_tensor(self,weight):
        freq_domain = torch.fft.fft2(weight)
        # freq_domain_shifted = torch.fft.fftshift(freq_domain)
        test_domain_shifted = torch.fft.ifft2(freq_domain)
        H, W = freq_domain.shape
        low_radius = min(H, W) // 10  
        high_radius = min(H, W)*self.high_ratio_for_fourier
        center = (H // 2, W // 2)
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist_from_center = torch.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        low_freq_mask = (dist_from_center <= low_radius).float()
        high_freq_mask = (dist_from_center > high_radius).float()

        band_pass_mask = ((dist_from_center >= low_radius) & (dist_from_center <= high_radius)).float()
        low_freq_component = freq_domain * low_freq_mask
        high_freq_component = freq_domain * high_freq_mask
        band_pass_component = freq_domain * band_pass_mask

        low_freq_spatial = torch.fft.ifft2(low_freq_component).real
        high_freq_spatial = torch.fft.ifft2(high_freq_component).real
        band_pass_spatial = torch.fft.ifft2(band_pass_component).real

        return low_freq_spatial,high_freq_spatial,band_pass_spatial
    
    def cal_low_and_high_fft_for_tensor_1D(self,weight):
        x_fft = torch.fft.fft(weight)
        n = x_fft.size(0)
        low_freq_idx = n // 8
        high_freq_idx = 2 * n // 3
        low_freq = torch.zeros_like(x_fft)
        low_freq[:low_freq_idx] = x_fft[:low_freq_idx]
        low_freq[-low_freq_idx:] = x_fft[-low_freq_idx:]
        mid_freq = torch.zeros_like(x_fft)
        mid_freq[low_freq_idx:high_freq_idx] = x_fft[low_freq_idx:high_freq_idx]
        high_freq = torch.zeros_like(x_fft)
        high_freq[high_freq_idx:] = x_fft[high_freq_idx:]
        high_freq[:n - high_freq_idx] = x_fft[:n - high_freq_idx]

        low_freq_spatial = torch.fft.ifft(low_freq).real
        band_pass_spatial = torch.fft.ifft(mid_freq).real
        high_freq_spatial = torch.fft.ifft(high_freq).real

        return low_freq_spatial,high_freq_spatial,band_pass_spatial
    
    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        # assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
                # if 'embeddings' not in param_name:
                #     new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
                # else:
                #     new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name]
        return TaskVectorFFT(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:   
                # merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]
                # if 'embedding' not in param_name and 'LayerNorm' not in param_name and 'bias' not in param_name:
                if 'weight' in param_name and pretrained_param_dict[param_name].ndim==2:
                    merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]
                else:
                    merged_params[param_name] = pretrained_param_dict[param_name] 
        return merged_params
    

class TaskVectorBREAD:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None,top_k_keep=0.1,top_k_remove=0.0):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        self.top_k_keep=top_k_keep
        self.top_k_remove=top_k_remove
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = self.mask_keep_top(self.mask_remove_top(self.task_vector_param_dict[param_name]))
    
    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        # assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVectorBREAD(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]

        return merged_params
    
    def mask_keep_top(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 0:
            return tensor
        else:
            top_k_int = int(tensor.shape[-1] * self.top_k_keep)
            _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
            mask = torch.zeros(tensor.shape)
            mask.scatter_(len(tensor.shape) - 1, masked_indices, 1)

            return mask * tensor

    def mask_remove_top(self, tensor: torch.Tensor) -> torch.Tensor:
        if len(tensor.shape) == 0:
            return tensor
        else:
            top_k_int = int(tensor.shape[-1] * self.top_k_remove)
            _, masked_indices = torch.topk(torch.abs(tensor), top_k_int)
            mask = torch.ones(tensor.shape)
            mask.scatter_(len(tensor.shape) - 1, masked_indices, 0.0)

            return mask * tensor
    
class TaskVectorTWIN:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None,top_k_keep=0.1,top_k_remove=0.0):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        self.top_k_keep=top_k_keep
        self.top_k_remove=top_k_remove
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]
                for param_name in param_names_to_merge:
                    self.task_vector_param_dict[param_name] = self.svd(self.task_vector_param_dict[param_name],top_k_keep)
    
    def svd(self,tensor: torch.Tensor, density: float,**kwargs,):
        if density >= 1:
            return tensor
        if density <= 0:
            return torch.zeros_like(tensor)
        if len(tensor.shape) <= 1:
            # rank=1
            return tensor
        driver = None
        if tensor.is_cuda:
            driver = 'gesvda'
        U, S, Vh = torch.linalg.svd(tensor, full_matrices=True, driver=driver)
        new_rank = int(density * len(S))
        U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]
        res = U @ torch.diag(S) @ Vh
        return res

    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        # assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVectorTWIN(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]
        return merged_params



