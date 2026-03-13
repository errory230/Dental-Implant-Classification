import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model.model import ImplantConvNeXtTinyModel
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inference for Dental Implant Classification")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--weight_path", type=str, default="weights/model_weight.pth", help="Path to model weight (.pth)")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes (closed set)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImplantConvNeXtTinyModel(num_classes=args.num_classes).to(device)
    
    try:
        checkpoint = torch.load(args.weight_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded weights from {args.weight_path}")
    except FileNotFoundError:
        print(f"Error: Could not find weight file at {args.weight_path}")
        return

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(args.image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device) 

        with torch.no_grad():
            logits, features = model(input_tensor)
            
            probs = F.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        print("-" * 30)
        print(f"Image: {args.image_path}")
        print(f"Prediction: Class {pred.item()}")
        print("-" * 30)
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    main()
