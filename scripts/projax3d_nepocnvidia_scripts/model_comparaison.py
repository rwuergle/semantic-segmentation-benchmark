from projax3d.tiling import GridCuboidSampler
from projax3d.interop.torch import SceneLoader, build_model, CombinedFocalLovaszLoss
from projax3d.interop.common import get_feature_config
from projax3d.interop.common.feature_config import FeatureConfig, op_one_hot, op_passthrough
from projax3d.io import Scene

import torch.nn.functional as F
from projax3d.interop.torch import compute_metrics, save_from_training
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import re
from pathlib import Path
import argparse

SITN_CLASS_MAP = {
    1: 0,
    2: 1,
    3: 2,
    6: 3,
    7: 0,
    9: 0,
    11: 0,
    14: 0,
    15: 0,
    17: 0,
    19: 0,
    21: 4,
    22: 5,
    25: 0,
    26: 6,
    29: 0,
}

SITN_NUM_CLASSES = 7 
DEVICE = "cuda:0"

def train_model_minkunet(train_loader, val_loader, model, prefix, epochs, base_dir = Path("./model_evaluations")):

    next_id = 0
    if base_dir.exists():
        existing_runs = []
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                match = re.search(rf"{prefix}_(\d+)", d.name)
                if match:
                    existing_runs.append(int(match.group(1)))
        
        if existing_runs:
            next_id = max(existing_runs) + 1

    run_name = f"{prefix}_{next_id:03d}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=epochs, pct_start=0.04)
    loss_fn = CombinedFocalLovaszLoss(ignore_index=-1)

    best_miou = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = len(train_loader)
        for batch in train_loader:
            pos, x, y, bidx = batch["pos"], batch["x"], batch["y"], batch["batch"]
            optimizer.zero_grad(set_to_none=True)
            logits = model(pos, bidx, x)          # fp32 — NO autocast for spconv
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            num_batches += 1
            epoch_train_loss += loss.item()
            optimizer.step()

        writer.add_scalar("Params/Learning_Rate", scheduler.get_last_lr()[0], epoch)
        scheduler.step()

        avg_train_loss = epoch_train_loss / num_batches
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)

        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = len(val_loader)
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                pos, x, y, bidx = batch["pos"], batch["x"], batch["y"], batch["batch"]
                logits = model(pos, bidx, x)

                val_loss = loss_fn(logits, y)
                epoch_val_loss += val_loss.item()

                m = y != -1
                preds.append(logits.argmax(-1)[m].cpu())
                labels.append(y[m].cpu())

        avg_val_loss = epoch_val_loss / num_val_batches
        metrics = compute_metrics(torch.cat(preds).numpy(),
                                torch.cat(labels).numpy(), SITN_NUM_CLASSES)
        print(f"E{epoch:03d} mIoU={metrics['miou']:.4f} loss_val={avg_val_loss:.4f} OA={metrics['oa']:.4f}")

        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Metrics/mIoU", metrics["miou"], epoch)
        writer.add_scalar("Metrics/OA", metrics["oa"], epoch)

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            save_from_training(f"{prefix}_small_weights.pt", model=model, loader=train_loader,
                            architecture=f"{prefix}",
                            training_info={"epoch": epoch, "miou": best_miou})
            
    writer.close()

def train_model_minkunet2(train_loader, val_loader, model, prefix, epochs, base_dir = Path("./model_evaluations")):

    next_id = 0
    if base_dir.exists():
        existing_runs = []
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                match = re.search(rf"{prefix}_(\d+)", d.name)
                if match:
                    existing_runs.append(int(match.group(1)))
        
        if existing_runs:
            next_id = max(existing_runs) + 1

    run_name = f"{prefix}_{next_id:03d}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=epochs, pct_start=0.04)
    loss_fn = CombinedFocalLovaszLoss(ignore_index=-1, gamma=2.0)

    best_miou = 0.0
    num_batches = len(train_loader)
    num_val_batches = len(val_loader)
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            pos, x, y, bidx = batch["pos"], batch["x"], batch["y"], batch["batch"]
            optimizer.zero_grad(set_to_none=True)
            logits = model(pos, bidx, x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            num_batches += 1
            epoch_train_loss += loss.item()
            optimizer.step()
            
        writer.add_scalar("Params/Learning_Rate", scheduler.get_last_lr()[0], epoch)
        scheduler.step()
        avg_train_loss = epoch_train_loss / num_batches
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        model.eval()
        epoch_val_loss = 0.0
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                pos, x, y, bidx = batch["pos"], batch["x"], batch["y"], batch["batch"]
                logits = model(pos, bidx, x)

                val_loss = loss_fn(logits, y)
                epoch_val_loss += val_loss.item()

                m = y != -1
                preds.append(logits.argmax(-1)[m].cpu())
                labels.append(y[m].cpu())

        avg_val_loss = epoch_val_loss / num_val_batches
        metrics = compute_metrics(torch.cat(preds).numpy(),
                                torch.cat(labels).numpy(), SITN_NUM_CLASSES)

        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Metrics/mIoU", metrics["miou"], epoch)
        writer.add_scalar("Metrics/OA", metrics["oa"], epoch)

        print(f"E{epoch:03d} mIoU={metrics['miou']:.4f} loss_val={avg_val_loss:.4f} OA={metrics['oa']:.4f}")

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            save_from_training(f"{prefix}_weights_model.pt", model=model, loader=train_loader,
                            architecture=f"{prefix}",
                            training_info={"epoch": epoch, "miou": best_miou})
            
    writer.close()

def train_model_minkunet3(train_loader, val_loader, model, prefix, epochs, base_dir = Path("./model_evaluations")):

    next_id = 0
    if base_dir.exists():
        existing_runs = []
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                match = re.search(rf"{prefix}_(\d+)", d.name)
                if match:
                    existing_runs.append(int(match.group(1)))
        
        if existing_runs:
            next_id = max(existing_runs) + 1

    run_name = f"{prefix}_{next_id:03d}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=3e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,eta_min=1e-7)
    loss_fn = CombinedFocalLovaszLoss(ignore_index=-1, gamma=2.0)

    best_miou = 0.0
    num_batches = len(train_loader)
    num_val_batches = len(val_loader)
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            pos, x, y, bidx = batch["pos"], batch["x"], batch["y"], batch["batch"]
            optimizer.zero_grad(set_to_none=True)
            logits = model(pos, bidx, x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            epoch_train_loss += loss.item()
            optimizer.step()
            
        writer.add_scalar("Params/Learning_Rate", scheduler.get_last_lr()[0], epoch)
        scheduler.step()
        avg_train_loss = epoch_train_loss / num_batches
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        model.eval()
        epoch_val_loss = 0.0
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                pos, x, y, bidx = batch["pos"], batch["x"], batch["y"], batch["batch"]
                logits = model(pos, bidx, x)

                val_loss = loss_fn(logits, y)
                epoch_val_loss += val_loss.item()

                m = y != -1
                preds.append(logits.argmax(-1)[m].cpu())
                labels.append(y[m].cpu())

        avg_val_loss = epoch_val_loss / num_val_batches
        metrics = compute_metrics(torch.cat(preds).numpy(),
                                torch.cat(labels).numpy(), SITN_NUM_CLASSES)

        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Metrics/mIoU", metrics["miou"], epoch)
        writer.add_scalar("Metrics/OA", metrics["oa"], epoch)

        print(f"E{epoch:03d} mIoU={metrics['miou']:.4f} loss_val={avg_val_loss:.4f} OA={metrics['oa']:.4f}")

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            save_from_training(f"{prefix}_v3_weights.pt", model=model, loader=train_loader,
                            architecture=f"{prefix}",
                            training_info={"epoch": epoch, "miou": best_miou})
            
    writer.close()

def train_model_minkunet2_pretrained(train_loader, val_loader, model, prefix, epochs, base_dir = Path("./model_evaluations")):

    next_id = 0
    if base_dir.exists():
        existing_runs = []
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                match = re.search(rf"{prefix}_(\d+)", d.name)
                if match:
                    existing_runs.append(int(match.group(1)))
        
        if existing_runs:
            next_id = max(existing_runs) + 1

    run_name = f"{prefix}_{next_id:03d}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = CombinedFocalLovaszLoss(ignore_index=-1, gamma=2.0)

    best_miou = 0.0
    num_batches = len(train_loader)
    num_val_batches = len(val_loader)
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            pos, x, y, bidx = batch["pos"], batch["x"], batch["y"], batch["batch"]
            optimizer.zero_grad(set_to_none=True)
            logits = model(pos, bidx, x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            num_batches += 1
            epoch_train_loss += loss.item()
            optimizer.step()
            
        writer.add_scalar("Params/Learning_Rate", scheduler.get_last_lr()[0], epoch)
        scheduler.step()
        avg_train_loss = epoch_train_loss / num_batches
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        model.eval()
        epoch_val_loss = 0.0
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                pos, x, y, bidx = batch["pos"], batch["x"], batch["y"], batch["batch"]
                logits = model(pos, bidx, x)

                val_loss = loss_fn(logits, y)
                epoch_val_loss += val_loss.item()

                m = y != -1
                preds.append(logits.argmax(-1)[m].cpu())
                labels.append(y[m].cpu())

        avg_val_loss = epoch_val_loss / num_val_batches
        metrics = compute_metrics(torch.cat(preds).numpy(),
                                torch.cat(labels).numpy(), SITN_NUM_CLASSES)

        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Metrics/mIoU", metrics["miou"], epoch)
        writer.add_scalar("Metrics/OA", metrics["oa"], epoch)

        print(f"E{epoch:03d} mIoU={metrics['miou']:.4f} loss_val={avg_val_loss:.4f} OA={metrics['oa']:.4f}")

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            save_from_training(f"{prefix}_weights_secondary_trained.pt", model=model, loader=train_loader,
                            architecture=f"{prefix}",
                            training_info={"epoch": epoch, "miou": best_miou})
            
    writer.close()

def train_model_ptv3(train_loader, val_loader, model, prefix, epochs, base_dir = Path("./model_evaluations")):

    next_id = 0
    if base_dir.exists():
        existing_runs = []
        for d in base_dir.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                match = re.search(rf"{prefix}_(\d+)", d.name)
                if match:
                    existing_runs.append(int(match.group(1)))
        
        if existing_runs:
            next_id = max(existing_runs) + 1

    run_name = f"{prefix}_{next_id:03d}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=5e-3, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,)
    loss_fn = CombinedFocalLovaszLoss(ignore_index=-1, gamma=2.0)

    best_miou = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = len(train_loader)
        for batch in train_loader:
            pos, x, y, bidx = batch["pos"], batch["x"], batch["y"], batch["batch"]
            optimizer.zero_grad(set_to_none=True)
            logits = model(pos, bidx, x)
            loss = loss_fn(logits, y)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=3.0
            )

            optimizer.step()
            epoch_train_loss += loss.item()

            
        writer.add_scalar("Params/Learning_Rate", scheduler.get_last_lr()[0], epoch)
        scheduler.step()

        avg_train_loss = epoch_train_loss / num_batches
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)

        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = len(val_loader)
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                pos, x, y, bidx = batch["pos"], batch["x"], batch["y"], batch["batch"]
                logits = model(pos, bidx, x)
                val_loss = loss_fn(logits, y)

                epoch_val_loss += val_loss.item()

                m = y != -1
                preds.append(logits.argmax(-1)[m].cpu())
                labels.append(y[m].cpu())

        avg_val_loss = epoch_val_loss / num_val_batches
        metrics = compute_metrics(torch.cat(preds).numpy(),
                                torch.cat(labels).numpy(), SITN_NUM_CLASSES)
        print(f"E{epoch:03d} mIoU={metrics['miou']:.4f} loss_val={avg_val_loss:.4f} OA={metrics['oa']:.4f}")

        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Metrics/mIoU", metrics["miou"], epoch)
        writer.add_scalar("Metrics/OA", metrics["oa"], epoch)

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            save_from_training(f"{prefix}_weights_v100.pt", model=model, loader=train_loader,
                            architecture=f"{prefix}",
                            training_info={"epoch": epoch, "miou": best_miou})
            
    writer.close()

def make_loader_minkunet(scene, shuffle, voxel_size=0.50, roi_cache_max_gb=50):
    scene.compute_footprint()
    sampler = GridCuboidSampler(scene, step=(50, 50), size=(15, 15), footprint="auto")
    return SceneLoader(
        scene, sampler, feature_config,
        batch_size=12,
        n_points=32768,
        class_map=SITN_CLASS_MAP, 
        label_column="classification",
        ignore_index=-1,
        shuffle=shuffle,
        #prefetch_factor = 0,
        streaming=True,
        pin_memory=True,
        num_workers=0,
        sample_policy="voxel",
        sample_policy_kwargs={"voxel_size": voxel_size},
        roi_cache_dir="auto",
        roi_cache_max_gb = roi_cache_max_gb,
    ).to_device(DEVICE)

def make_loader_minkunetv2(scene, shuffle, voxel_size=0.25, roi_cache_max_gb = 50):
    sampler = GridCuboidSampler(scene, step=(40, 40), size=(40, 40), footprint="auto")
    return SceneLoader(
        scene, sampler, feature_config,
        batch_size=8,
        n_points=250_000,
        class_map=SITN_CLASS_MAP, 
        label_column="classification",
        ignore_index=-1,
        shuffle=shuffle,
        streaming=True,
        pin_memory=True,
        num_workers=0,
        sample_policy="voxel",
        sample_policy_kwargs={"voxel_size": voxel_size},
        roi_cache_dir="auto",
        roi_cache_max_gb = roi_cache_max_gb,
    ).to_device(DEVICE)

def make_loader_ptv3(scene, shuffle, roi_cache_max_gb = 50):
    sampler = GridCuboidSampler(scene, step=(40, 40), size=(40, 40), footprint="auto")
    return SceneLoader(
        scene, sampler, feature_config,
        batch_size=2,
        n_points=250_000,
        class_map=SITN_CLASS_MAP, 
        label_column="classification",
        ignore_index=-1,
        shuffle=shuffle,
        streaming=True,
        num_workers=0,
        roi_cache_dir="auto",
        roi_cache_max_gb = roi_cache_max_gb,
    ).to_device(DEVICE)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Deep learning workflow")
    parser.add_argument("--architecture", type=str, default="minkunetv3", help="name of the architecture to train")
    args = parser.parse_args()

    scene_train = Scene.from_parquet(r"L:\raphael\projax-3d-models\data\Train", name="train", crs="EPSG:2056")
    scene_train.compute_footprint()
    scene_train.set_cache_budget(70 * 1024**3)
    scene_validation = Scene.from_parquet(r"L:\raphael\projax-3d-models\data\Test", name="validation", crs="EPSG:2056")
    scene_validation.compute_footprint()
    scene_validation.set_cache_budget(20 * 1024**3)

    if args.architecture == "minkunet":
        print(f"{'Training MinkUnet':=^50}")

        VOXEL = 0.25
        EPOCHS = 100
        prefix = args.architecture

        feature_config = get_feature_config("lidar_standard")
        train_loader = make_loader_minkunet(scene_train, shuffle=True, voxel_size=VOXEL, roi_cache_max_gb=150)
        val_loader   = make_loader_minkunet(scene_validation, shuffle=False, voxel_size=VOXEL, roi_cache_max_gb=50)

        model = build_model(args.architecture, {
            "num_classes": SITN_NUM_CLASSES,
            "in_channels": feature_config.num_features,
            "grid_size": VOXEL,
            "channels": (8, 16, 32, 64),
        }).to(DEVICE)

        train_model_minkunet(train_loader, val_loader, model, prefix, EPOCHS)
    
    if args.architecture == "minkunetv2":
        args.architecture = "minkunet"
        print(f"{'Training MinkUnet v2':=^50}")

        VOXEL = 0.25
        EPOCHS = 100
        prefix = args.architecture

        feature_config = get_feature_config("lidar_standard")
        train_loader = make_loader_minkunetv2(scene_train, shuffle=True, voxel_size=VOXEL, roi_cache_max_gb=150)
        val_loader   = make_loader_minkunetv2(scene_validation, shuffle=False, voxel_size=VOXEL, roi_cache_max_gb=150)

        model = build_model(args.architecture, {
            "num_classes": SITN_NUM_CLASSES,
            "in_channels": feature_config.num_features,
            "grid_size": VOXEL,
            "channels": (96, 192, 384, 768),
        }).to(DEVICE)

        train_model_minkunet2(train_loader, val_loader, model, prefix, EPOCHS)

    if args.architecture == "minkunetv3":
        args.architecture = "minkunet"
        print(f"{'Training MinkUnet v3':=^50}")

        VOXEL = 0.25
        EPOCHS = 100
        prefix = args.architecture

        feature_config = get_feature_config("lidar_standard")
        train_loader = make_loader_minkunetv2(scene_train, shuffle=True, voxel_size=VOXEL, roi_cache_max_gb=150)
        val_loader   = make_loader_minkunetv2(scene_validation, shuffle=False, voxel_size=VOXEL, roi_cache_max_gb=50)

        model = build_model(args.architecture, {
            "num_classes": SITN_NUM_CLASSES,
            "in_channels": feature_config.num_features,
            "grid_size": VOXEL,
            "channels": (128, 256, 512, 1024),
        }).to(DEVICE)

        train_model_minkunet3(train_loader, val_loader, model, prefix, EPOCHS)

    if args.architecture == "minkunet_pretrained":
        args.architecture = "minkunet"
        print(f"{'Training MinkUnet pretrained':=^50}")

        VOXEL = 0.25
        EPOCHS = 50
        prefix = args.architecture

        feature_config = get_feature_config("lidar_standard")
        train_loader = make_loader_minkunetv2(scene_train, shuffle=True, voxel_size=VOXEL, roi_cache_max_gb=150)
        val_loader   = make_loader_minkunetv2(scene_validation, shuffle=False, voxel_size=VOXEL, roi_cache_max_gb=150)

        checkpoint = torch.load(r"L:\raphael\projax-3d-models\minkunet_weights_secondary_trained.pt", map_location=DEVICE)

        model = build_model(args.architecture, {
            "num_classes": SITN_NUM_CLASSES,
            "in_channels": feature_config.num_features,
            "grid_size": VOXEL,
            "channels": (96, 192, 384, 768),
        })
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)

        train_model_minkunet2_pretrained(train_loader, val_loader, model, prefix, EPOCHS)

    if args.architecture == "ptv3_seg":
        VOXEL = 0.5
        EPOCHS = 100
        prefix = args.architecture

        feature_config = get_feature_config("lidar_ptv3")
        train_loader = make_loader_ptv3(scene_train, shuffle=True, roi_cache_max_gb = 150)
        val_loader   = make_loader_ptv3(scene_validation, shuffle=False, roi_cache_max_gb = 50)

        model = build_model(args.architecture, {
            "num_classes": SITN_NUM_CLASSES,
            "in_channels": feature_config.num_features,
            "grid_size": VOXEL,

            "backbone_out_channels": 256,

            "backbone_config": {
                "enc_depths": (2, 3, 6, 8, 3),
                "enc_channels": (96, 192, 384, 768, 768),
                "enc_num_head": (6, 12, 24, 48, 48),

                "dec_depths": (3, 3, 3, 3),
                "dec_channels": (256, 256, 384, 768),
                "dec_num_head": (8, 8, 12, 24),

                "enc_patch_size": (256,256,256,256,256),
                "dec_patch_size": (256,256,256,256),

                "mlp_ratio": 4,
                "drop_path": 0.2,
                "enable_flash": True,
            }
        }).to(DEVICE)

        train_model_ptv3(train_loader, val_loader, model, prefix, EPOCHS)

    if args.architecture == "kpconv_seg":
        EPOCHS = 100
        prefix = args.architecture

        feature_config = get_feature_config("lidar_standard")
        train_loader = make_loader_kpconv(scene_train, shuffle=True, roi_cache_max_gb=150)
        val_loader   = make_loader_kpconv(scene_validation, shuffle=False, roi_cache_max_gb = 50)

        model = build_model(args.architecture, {
            "num_classes": SITN_NUM_CLASSES,
            "in_channels": feature_config.num_features,
            "first_features_dim": 128
        }).to(DEVICE)

        train_model_kpconv(train_loader, val_loader, model, prefix, EPOCHS)

    if args.architecture == "kpconv_seg_second_block":
        args.architecture = "kpconv_seg"
        EPOCHS = 100
        prefix = args.architecture

        feature_config = FeatureConfig(
            source_columns=['minkunet_prediction', "entropy"],
            features={
                'minkunet_prediction': (['minkunet_prediction'], op_one_hot(SITN_NUM_CLASSES, 0)),
                "entropy": (['entropy'], op_passthrough())
            }
        )

        train_loader = make_loader_kpconv(scene_train, shuffle=True, roi_cache_max_gb=150)
        val_loader   = make_loader_kpconv(scene_validation, shuffle=False, roi_cache_max_gb = 50)

        model = build_model(args.architecture, {
            "num_classes": SITN_NUM_CLASSES,
            "in_channels": feature_config.num_features,
        }).to(DEVICE)

        train_model_kpconv(train_loader, val_loader, model, prefix, EPOCHS)
    