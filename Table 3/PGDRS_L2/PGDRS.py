import torch
import torch.nn.functional as F
import copy
import numpy as np
from PIL import Image
from attack import Attack
import torchvision.transforms as transforms
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from budget_l2 import BUDGET, project_rgb_uint8
from torchvision.models import convnext_tiny


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def predict_image_from_rgb_matrices(r_matrix, g_matrix, b_matrix, index_path, weight_path, index, model_cnn):
    img_array = np.stack([r_matrix, g_matrix, b_matrix], axis=-1).astype(np.uint8)
    img = Image.fromarray(img_array)
    img_tensor = data_transform(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_tensor.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    predicted_class_index = torch.argmax(classification_probability).item()

    return predicted_class_index, output[index].item()

def con_transform(actual_image_transform_matrix, adversarial_sample_transform_matrix, actual_image_matrix):
    adv_image = actual_image_matrix.copy()
    adversarial_sample_transform_matrix = adversarial_sample_transform_matrix.cpu().numpy()
    factors = np.array([[0.229, 0.485], [0.224, 0.456], [0.225, 0.406]])
    scales = np.array([255, 255, 255])

    for c in range(3):
        actual_transform = actual_image_transform_matrix[c]
        adversarial_transform = adversarial_sample_transform_matrix[c]
        actual_image = actual_image_matrix[c]

        mask_greater = adversarial_transform > actual_transform
        mask_less = adversarial_transform < actual_transform
        mask_equal = adversarial_transform == actual_transform

        adv_image[c] = np.where(mask_greater,
                                np.ceil((adversarial_transform * factors[c][0] + factors[c][1]) * scales[c]),
                                np.where(mask_less,
                                         np.floor((adversarial_transform * factors[c][0] + factors[c][1]) * scales[c]),
                                         np.where(mask_equal, actual_image, adv_image[c])))

    adv_image = np.clip(adv_image, 0, 255)

    return adv_image

class Noise:
    def __init__(self, noise_type, noise_sd):
        self.noise_type = noise_type
        self.noise_sd = noise_sd

    def __call__(self, img):
        if self.noise_type == "guassian":
            noise = torch.randn_like(img.float()) * self.noise_sd
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(img.float()) - 0.5) * 2 * self.noise_sd
        return noise


class PGDRSL2(Attack):
    def __init__(
        self,
        model,
        eps=BUDGET,
        alpha=0.001,
        steps=int(8 / 255 * (2.4285 - (-2.0357)) / 0.001),
        noise_type="guassian",
        noise_sd=0.15,
        noise_batch_size=5,
        batch_max=2048,
        eps_for_division=1e-10,
    ):
        super().__init__("PGDRSL2", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.noise_func = Noise(noise_type, noise_sd)
        self.noise_batch_size = noise_batch_size
        self.eps_for_division = eps_for_division
        self.supported_mode = ["default", "targeted"]
        self.batch_max = batch_max

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, actual_image_transform_matrix, actual_image_matrix) -> torch.Tensor:
        if inputs.shape[0] * self.noise_batch_size > self.batch_max:
            img_list = []
            split_num = int(self.batch_max / self.noise_batch_size)
            inputs_split = torch.split(inputs, split_size_or_sections=split_num)
            labels_split = torch.split(labels, split_size_or_sections=split_num)
            for img_sub, lab_sub in zip(inputs_split, labels_split):
                img_adv = self._forward(img_sub, lab_sub, actual_image_transform_matrix, actual_image_matrix)
                img_list.append(img_adv)
            return torch.vstack(img_list)
        else:
            return self._forward(inputs, labels, actual_image_transform_matrix, actual_image_matrix)

    def _forward(self, images, labels, actual_image_transform_matrix, actual_image_matrix):
        r"""
        Overridden.
        """
        weights_path = r"..."
        json_absolute_path = r"..."

        model_current = convnext_tiny

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        # expend the inputs over noise_batch_size
        shape = (
            torch.Size([images.shape[0], self.noise_batch_size]) + images.shape[1:]
        )  # nopep8
        inputs_exp = images.unsqueeze(1).expand(shape)
        inputs_exp = inputs_exp.reshape(
            torch.Size([-1]) + inputs_exp.shape[2:]
        )  # nopep8

        data_batch_size = labels.shape[0]
        delta = torch.zeros(
            (len(labels), *inputs_exp.shape[1:]), requires_grad=True, device=self.device
        )
        delta_last = torch.zeros(
            (len(labels), *inputs_exp.shape[1:]),
            requires_grad=False,
            device=self.device,
        )

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        for _ in range(self.steps):
            print(_)
            delta.requires_grad = True
            # img_adv is the perturbed data for randmized smoothing
            # delta.repeat(1,self.noise_batch_size,1,1)
            img_adv = inputs_exp + delta.unsqueeze(1).repeat(
                (1, self.noise_batch_size, 1, 1, 1)
            ).view_as(
                inputs_exp
            )  # nopep8
            img_adv = torch.clamp(img_adv, min=-10, max=10)

            noise_added = self.noise_func(img_adv.view(len(img_adv), -1))
            noise_added = noise_added.view(img_adv.shape)

            adv_noise = img_adv + noise_added
            logits = self.get_logits(adv_noise)

            softmax = F.softmax(logits, dim=1)
            # average the probabilities across noise
            average_softmax = (
                softmax.reshape(-1, self.noise_batch_size, logits.shape[-1])
                .mean(1, keepdim=True)
                .squeeze(1)
            )
            logsoftmax = torch.log(average_softmax.clamp(min=1e-20))
            ce_loss = (
                F.nll_loss(logsoftmax, labels)
                if not self.targeted
                else -F.nll_loss(logsoftmax, target_labels)
            )

            grad = torch.autograd.grad(
                ce_loss, delta, retain_graph=False, create_graph=False
            )[0]
            grad_norms = (
                torch.norm(grad.view(data_batch_size, -1), p=2, dim=1)
                + self.eps_for_division
            )
            grad = grad / grad_norms.view(data_batch_size, 1, 1, 1)

            delta = delta_last + self.alpha * grad.sign()
            delta_norms = torch.norm(delta.view(data_batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)
            delta_last.data = copy.deepcopy(delta.data)


            iterative_image = images.clone()
            iterative_delta = delta.clone()
            adv_images = torch.clamp(iterative_image + iterative_delta, min=-10, max=10).detach()
            adv_images[0][0] = torch.clamp(iterative_image[0][0] + iterative_delta[0][0], min=-2.1179, max=2.2489).detach()
            adv_images[0][1] = torch.clamp(iterative_image[0][1] + iterative_delta[0][1], min=-2.0357, max=2.4285).detach()
            adv_images[0][2] = torch.clamp(iterative_image[0][2] + iterative_delta[0][2], min=-1.8044, max=2.64).detach()

            adv_images = adv_images.squeeze(0).cpu()
            adv_image = con_transform(actual_image_transform_matrix, adv_images, actual_image_matrix)
            adv_image = project_rgb_uint8(adv_image.transpose(1, 2, 0), actual_image_matrix.transpose(1, 2, 0), self.eps).transpose(2, 0, 1)

            iterative_image_top1_label, x = predict_image_from_rgb_matrices(adv_image[0], adv_image[1], adv_image[2], json_absolute_path, weights_path, labels.cpu(), model_current)
            if iterative_image_top1_label != labels:
                return adv_image

        return adv_image
