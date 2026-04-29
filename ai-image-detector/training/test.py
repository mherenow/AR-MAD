import sys
import os
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classify_image import load_model

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          '..', 'checkpoints', 'all_features', 'best_model.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(CHECKPOINT, device)
model.eval()

# Standard ImageNet preprocessing - must match training exactly
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

for path, label in [('datasets/synthbuster/dalle3/r0ab86ea2t.png', 'FAKE'), ('datasets/coco2017/train2017/000000000201.jpg', 'REAL')]:
    if not os.path.exists(path):
        print(f"Skipping {path} (not found)")
        continue
    img = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        logit = out[0] if isinstance(out, tuple) else out

    prob = torch.sigmoid(logit).item()
    print(f"{label}: logit={logit.item():.4f}, prob={prob:.4f}, "
          f"pred={'FAKE' if logit.item() > 0 else 'REAL'}")
