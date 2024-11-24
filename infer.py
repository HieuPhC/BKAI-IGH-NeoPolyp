import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import segmentation_models_pytorch as smp
import argparse

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Infer a segmentation model on an input image.")
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True, 
        help="Path to the input image for inference."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="output_result.jpeg", 
        help="Path to save the resulting segmentation mask (default: output_result.jpeg)."
    )
    return parser.parse_args()

# Function to convert mask to RGB
def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, color in color_dict.items():
        output[mask == k] = color
    return output

def main(image_path, output_path):
    # Hyperparameters and paths
    learning_rate = 0.0001
    save_path = 'model.pth'

    # Define model
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    )

    # Load model and optimizer states
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    checkpoint = torch.load(save_path, weights_only=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Validation transformation
    val_transformation = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Color dictionary
    color_dict = {
        0: (0, 0, 0),       # Background
        1: (255, 0, 0),     # Class 1
        2: (0, 255, 0)      # Class 2
    }

    # Load and preprocess the test image
    ori_img = cv2.imread(image_path)
    if ori_img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = ori_img.shape[:2]
    img = cv2.resize(ori_img, (256, 256))

    # Apply transformations
    transformed = val_transformation(image=img)
    input_img = transformed["image"].unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Post-process the output
    mask = cv2.resize(output_mask, (ori_w, ori_h))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask, color_dict)

    # Save the result
    mask_rgb_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(output_path, mask_rgb_bgr)

    if not success:
        raise RuntimeError(f"Failed to save the image to {output_path}")
    else:
        print(f"Image saved successfully at {output_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args.image_path, args.output_path)

