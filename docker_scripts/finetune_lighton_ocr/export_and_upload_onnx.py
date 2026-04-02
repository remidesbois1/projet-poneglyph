"""
Export manuel de LightOnOCR (Vision Encoder) vers ONNX via PyTorch brut.
Contourne totalement Optimum qui ne supporte pas cette architecture custom.

Usage :
  python export_and_upload_onnx.py
"""

import os
import sys
import torch
import shutil
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText

SOURCE_REPO = "Remidesbois/LightonOCR-2-1b-poneglyph"
TARGET_REPO = "Remidesbois/LightonOCR-2-1b-poneglyph-ONNX"
LOCAL_DIR   = "downloaded_model"
ONNX_DIR    = "onnx_output"

def main():
    load_dotenv(".env")
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN manquant dans le fichier .env")
        sys.exit(1)

    print(f"\n📥 1. Téléchargement de {SOURCE_REPO}...")
    snapshot_download(repo_id=SOURCE_REPO, local_dir=LOCAL_DIR, token=token)

    print("\n🧠 2. Chargement du modèle en RAM (PyTorch)...")
    processor = AutoProcessor.from_pretrained(LOCAL_DIR, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        LOCAL_DIR, 
        trust_remote_code=True, 
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    os.makedirs(ONNX_DIR, exist_ok=True)
    onnx_vision_path = os.path.join(ONNX_DIR, "vision_encoder.onnx")

    print("\n🔨 3. Création des Tenseurs factices (Dummy Inputs) via le Processor...")
    # On génère une image factice et on laisse le processor calculer la vraie forme (patchs, etc.) attendue par LightOn
    dummy_image = Image.new('RGB', (1000, 1000), color='white')
    # On ajoute le token d'image pad obligatoire pour que le processeur crée le bon nombre de "features"
    prompt = "<|image_pad|>\nTexte factice"
    
    inputs = processor(text=prompt, images=dummy_image, return_tensors="pt")
    dummy_pixel_values = inputs["pixel_values"].to(torch.float32)
    
    # S'il y a des tailles d'images requises (ex: pour Qwen-VL architecture)
    args_tuple = (dummy_pixel_values,)
    if "image_sizes" in inputs:
        args_tuple = (dummy_pixel_values, inputs["image_sizes"])

    print("🔍 Recherche de l'encodeur visuel dans l'architecture...")
    # Recherche dynamique de la sous-couche Vision (selon comment LightOn a imbriqué les classes)
    vision_model = None
    if hasattr(model, "vision_encoder"):
        vision_model = model.vision_encoder
    elif hasattr(model, "model") and hasattr(model.model, "vision_encoder"):
        vision_model = model.model.vision_encoder
    elif hasattr(model, "vision_tower"):
        vision_model = model.vision_tower
    elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vision_model = model.model.vision_tower

    if vision_model is None:
        print("❌ Impossible de trouver la couche 'vision_encoder' ou 'vision_tower'.")
        sys.exit(1)

    print("🚀 4. Lancement de torch.onnx.export (Contournement d'Optimum)...")
    try:
        # Export brut de la couche visuelle
        torch.onnx.export(
            vision_model,
            args_tuple,
            onnx_vision_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['pixel_values'] + (['image_sizes'] if len(args_tuple) > 1 else []),
            output_names=['vision_embeddings'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size', 1: 'num_patches_or_channels'},
                'vision_embeddings': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        print(f"✅ Export PyTorch réussi : {onnx_vision_path}")
    except Exception as e:
        print(f"❌ Échec de l'export PyTorch natif : {e}")
        sys.exit(1)

    print("\n📦 5. Préparation des fichiers de configuration...")
    # Copie des configs vitales pour que l'architecture soit reconnue
    config_files = [f for f in os.listdir(LOCAL_DIR) if f.endswith('.json') or f.endswith('.txt') or f.endswith('.jinja')]
    for f in config_files:
        shutil.copy2(os.path.join(LOCAL_DIR, f), os.path.join(ONNX_DIR, f))
        
    size_mb = os.path.getsize(onnx_vision_path) / (1024 * 1024)
    print(f"  📊 vision_encoder.onnx : {size_mb:.1f} MB")

    print(f"\n☁️ 6. Upload vers Hugging Face ({TARGET_REPO})...")
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=TARGET_REPO, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=ONNX_DIR,
            repo_id=TARGET_REPO,
            repo_type="model",
            path_in_repo="/"
        )
        print("\n🎉 TERMINÉ ! L'encodeur visuel est uploadé au format ONNX.")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'upload : {e}")

if __name__ == "__main__":
    main()