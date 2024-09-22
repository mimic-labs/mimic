import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from scipy.ndimage import binary_closing, binary_opening
import torchvision.transforms as transforms
from typing import Tuple


########################## SAM2/DinoV2 Initializations ##########################
def initialize_sam2(device):
    sam2_checkpoint = "../third_party/segment-anything-2/checkpoints/sam2_hiera_base_plus.pt"
    model_cfg = "sam2_hiera_b+.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

    mask_predictor = SAM2AutomaticMaskGenerator(sam2)
    prompted_predictor = SAM2ImagePredictor(sam2)
    return sam2, mask_predictor, prompted_predictor

def initialize_dinov2(device, REPO_NAME = "facebookresearch/dinov2", MODEL_NAME = "dinov2_vitb14"):
    model = torch.hub.load(repo_or_dir=REPO_NAME, model=MODEL_NAME)
    model.to(device)
    model.eval()
    return model
################################################################################


########################## SAM2 Helper Functions ##########################
def generate_masks(mask_generator, prompted_predictor, image_path, point_prompt=None, box_prompt=None):
    """
    mask_generator: output of initialize_sam2 for all mask generation
    prompted_predictor: output of initialize_sam2 for box and points prompted generation
    image_path: path of image
    point_prompt: np.array(n,2), where n is # of points
    box_prompt: np.array(n,4), where n is # of boxes
    """
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    if point_prompt is None and box_prompt is None:
        masks = mask_generator.generate(image)
        valid_indices = deduplicate_masks(masks)
        return [masks[i] for i in range(len(masks)) if i in valid_indices]
    else:
        prompted_predictor.set_image(image)
        masks, scores, logits = prompted_predictor.predict(
            point_coords=point_prompt,
            box=box_prompt,
            multimask_output=True,
            point_labels=np.array([1]*len(point_prompt)),
        )
        return masks, scores, logits

def deduplicate_masks(masks, threshold=0.7):
    valid = set(list(range(len(masks))))
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            mask1, mask2 = masks[i]['segmentation'], masks[j]['segmentation']
            iou = (mask1*mask2).sum()/(mask1+mask2).sum()
            if iou >= threshold:
                valid.remove(j)
    return valid
    
def create_mask_images(original_image, masks):
    """
    original_image: Image
    masks: np.array of each mask in original_image
    """
    original_image_array = np.array(original_image)
    all_cropped_imgs = []

    for i in range(masks.shape[0]):
        mask = masks[i, :, :]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        np_mask = mask.unsqueeze(dim=2).numpy()
        masked_img = (original_image_array * np_mask).astype(np.uint8)  # Convert to uint8
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            continue
        bbox = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        cropped_image_array = masked_img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
        cropped_image_pil = Image.fromarray(cropped_image_array)
        all_cropped_imgs.append(cropped_image_pil)

    return all_cropped_imgs
################################################################################


########################## DinoV2 Helper Functions ##########################
def make_transform(smaller_edge_size: int) -> transforms.Compose:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.BICUBIC

    return transforms.Compose([
        transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

def prepare_image(image: Image,
                  smaller_edge_size: float,
                  patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    transform = make_transform(int(smaller_edge_size))
    image_tensor = transform(image)

    # Crop image to dimensions that are a multiple of the patch size
    height, width = image_tensor.shape[1:] # C x H x W
    cropped_width, cropped_height = width - width % patch_size, height - height % patch_size
    image_tensor = image_tensor[:, :cropped_height, :cropped_width]

    grid_size = (cropped_height // patch_size, cropped_width // patch_size) # h x w (TODO: check)
    scale_width = image.width / cropped_width
    scale_height = image.height / cropped_height
    return image_tensor, grid_size, (scale_width, scale_height)

def get_obj_embeddings(model, cropped_obj):
    """
    model - dinov2 model
    cropped_obj: 1 Image
    """

    image_tensor, grid_size, scales = prepare_image(cropped_obj, 448, 14)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    tokens, cls = model.get_intermediate_layers(image_tensor.unsqueeze(0).to(device), return_class_token=True)[0]
    return tokens.detach().cpu().squeeze(), cls.detach().cpu().squeeze(), (grid_size, scales)
################################################################################

########################## General Helper Functions ##########################
def calculate_simmatrix(a, b, eps=1e-8):
    """
    a: NxD
    b: MxD
    
    out: NxM, each row contains how similar N_i is to each M
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = a_norm @ b_norm.T
    return sim_mt
    
def match_ref_and_query(query_embeddings, reference_embeddings):
    """
    query_embedding: NxD
    returns closest reference_embedding match to each query_embedding

    """
    similarities = calculate_simmatrix(query_embeddings, reference_embeddings)
    matched_ref_masks_idx = torch.argmax(similarities, dim=1)
    # self.matched_query_masks = self.query_masks[matched_query_masks_idx, :, :]
    # self.matched_query_patch_embeddings = self.query_patch_embeddings[matched_query_masks_idx, :, :]
    return matched_ref_masks_idx
################################################################################

def get_mask1_bestmask2(models, image_path1, image_path2, pos_points):
    """
    models = (dinov2, mask_predictor, prompted_predictor)
    
    pos_points: np.array(N,2) of points, generally it will look like np.array([[475,280]])

    Returns mask in image1 and best corresponding mask in image2
    """
    dinov2, mask_predictor, prompted_predictor = models
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Mask Generation
    masks1, scores, logits = generate_masks(mask_predictor, prompted_predictor, image_path1, point_prompt=pos_points) # point prompt for image1
    best_mask1 = masks1[[np.argmax(scores)]] # get best mask in image1
    masks2 = generate_masks(mask_predictor, prompted_predictor, image_path2) # get all masks in image2

    # PIL.Image Generation from masks
    cropped_images1 = create_mask_images(image1, best_mask1)
    cropped_images2 = create_mask_images(image2, np.array([mask['segmentation'] for mask in masks2]))

    # Calculate DinoV2 embeddings for all masks and get best match in image2
    query_cls = get_obj_embeddings(dinov2, cropped_images1[0])[1].unsqueeze(0) #get embedding for touched mask in image1; 1,768
    refs_cls = torch.stack([get_obj_embeddings(dinov2, c)[1] for c in cropped_images2]) #get embeddings for each mask in reference; Nx768
    idxs = match_ref_and_query(query_cls, refs_cls) # for each query, match to 1 reference
    return cropped_images1[0], cropped_images2[idxs[0].item()]