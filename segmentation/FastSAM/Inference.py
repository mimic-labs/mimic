import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()


def main(args):
    # load model
    model = FastSAM(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)
    input = Image.open(args.img_path)
    input = input.convert("RGB")
    everything_results = model(
        input,
        device=args.device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou    
        )
    bboxes = None
    points = None
    point_label = None
    prompt_process = FastSAMPrompt(input, everything_results, device=args.device)
    if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=args.box_prompt)
            bboxes = args.box_prompt
    elif args.text_prompt != None:
        ann = prompt_process.text_prompt(text=args.text_prompt)
    elif args.point_prompt[0] != [0, 0]:
        ann = prompt_process.point_prompt(
            points=args.point_prompt, pointlabel=args.point_label
        )
        points = args.point_prompt
        point_label = args.point_label
    else:
        ann = prompt_process.everything_prompt()
    prompt_process.plot(
        annotations=ann,
        output_path=args.output+args.img_path.split("/")[-1],
        bboxes = bboxes,
        points = points,
        point_label = point_label,
        withContours=args.withContours,
        better_quality=args.better_quality,
    )

def calculate_area(mask):
    return torch.sum(mask).item()

def calculate_intersection_over_min_area(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    intersection_area = calculate_area(intersection)
    mask1_area = calculate_area(mask1)
    mask2_area = calculate_area(mask2)
    min_area = min(mask1_area, mask2_area)
    intersection_over_min_area = intersection_area / min_area
    return intersection_over_min_area

def remove_overlapping_masks(masks, threshold=0.8):
    ignore_indices = set()

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            intersection_over_min_area = calculate_intersection_over_min_area(masks[i], masks[j])
            if intersection_over_min_area > threshold:
                if torch.sum(masks[i]) < torch.sum(masks[j]):
                    ignore_indices.add(i)
                else:
                    ignore_indices.add(j)

    filtered_masks = torch.stack([mask for idx, mask in enumerate(masks) if idx not in ignore_indices])
    return filtered_masks


def segment(img_path, point_prompt="[[0,0]]", box_prompt="[[0,0,0,0]]", text_prompt=None, point_label="[0]", conf=0.4, iou=0.9, output="./output/", filter=True):
    # load model
    model = FastSAM("../FastSAM.pt")
    point_prompt = ast.literal_eval(point_prompt)
    box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(box_prompt))
    point_label = ast.literal_eval(point_label)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    input = Image.open(img_path)
    input = input.convert("RGB")
    everything_results = model(
        input,
        device=device,
        retina_masks=True,
        imgsz=1024,
        conf=conf,
        iou=iou    
        )
    
    bboxes = None
    points = None
    # point_label = None
    prompt_process = FastSAMPrompt(input, everything_results, device=device)
    if box_prompt[0][2] != 0 and box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=box_prompt)
            bboxes = box_prompt
    elif text_prompt != None:
        ann = prompt_process.text_prompt(text=text_prompt)
    elif point_prompt[0] != [0, 0]:
        ann = prompt_process.point_prompt(
            points=point_prompt, pointlabel=point_label
        )
        points = point_prompt
        point_label = point_label
    else:
        ann = prompt_process.everything_prompt()

    if filter:
        filtered_ann = remove_overlapping_masks(ann)
    else:
        filtered_ann = ann

    prompt_process.plot(
        annotations=filtered_ann,
        output_path=output+img_path.split("/")[-1],
        bboxes = bboxes,
        points = points,
        point_label = point_label,
        withContours=False,
        better_quality=False,
    )

    return filtered_ann



if __name__ == "__main__":
    args = parse_args()
    main(args)
