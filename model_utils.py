import os, time, torch, shutil
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(num_classes=2, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_from_folder(folder, model_save_path, epochs=3, lr=1e-3, batch_size=8, status_callback=None):
    """Train a small model using torchvision.datasets.ImageFolder.
    Expects dataset organized as: folder/train/<class>/*.jpg and folder/val/<class>/*.jpg
    status_callback: function taking a dict to update status in the Flask app.
    """
    device = get_device()
    if status_callback:
        status_callback({'phase':'loading','progress':0,'last_message':'Loading dataset'})
    train_dir = os.path.join(folder, 'train')
    val_dir = os.path.join(folder, 'val')
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError('Expected dataset at uploads/dataset/train and uploads/dataset/val')

    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = get_model(num_classes=len(train_ds.classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running=0.0
        total=0
        for i,(x,y) in enumerate(train_loader,1):
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            total += 1
            if status_callback and i%10==0:
                status_callback({'phase':'training','progress': int(100*epoch/epochs), 'last_message':f'Epoch {epoch} batch {i} loss {loss.item():.3f}' })
        # validation
        model.eval()
        correct=0; n=0
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(device); y=y.to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                correct += (preds==y).sum().item()
                n += y.size(0)
        acc = correct / max(1,n)
        if status_callback:
            status_callback({'phase':'validation','progress': int(100*epoch/epochs), 'last_message': f'Epoch {epoch} val_acc {acc:.3f}'})
    # save model
    torch.save(model.state_dict(), model_save_path)
    return model_save_path

def load_image_for_infer(path):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    return transform(img).unsqueeze(0)

def infer_image(model, image_path, model_path=None):
    device = get_device()
    model.to(device)
    if model_path and os.path.isfile(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception:
            # ignore load errors, continue with random weights
            pass
    model.eval()
    x = load_image_for_infer(image_path).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs.tolist()

# simple grad-cam helper (not used directly in app by default)
def grad_cam(model, image_tensor, target_layer=None):
    model.eval()
    device = get_device()
    image_tensor = image_tensor.to(device)
    if target_layer is None:
        # try to use last conv layer for resnet
        for name,module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
    gradients = None; activations = None
    def save_grad(grad):
        nonlocal gradients; gradients = grad
    def forward_hook(module, inp, out):
        nonlocal activations; activations = out.detach()
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(lambda m,gi,go: save_grad(go[0]))
    out = model(image_tensor)
    idx = out.argmax(dim=1)
    out[0, idx].backward()
    pooled = torch.mean(gradients, dim=[0,2,3])
    act = activations[0]
    for i in range(act.shape[0]):
        act[i] *= pooled[i]
    heatmap = torch.mean(act, dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max()+1e-8)
    h1.remove(); h2.remove()
    return heatmap
