import argparse
import cv2
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from face_box import face_box
from model.recon import face_model
from util.preprocess import get_data_path
from util.io import visualize


def create_args(input_path, output_dir, device='cuda'):
    args = argparse.Namespace()
    
    args.inputpath = str(input_path)
    args.savepath = str(output_dir)
    args.device = device
    
    args.iscrop = True
    args.detector = 'retinaface'
    
    args.ldm68 = True
    args.ldm106 = True
    args.ldm106_2d = True
    args.ldm134 = True
    args.seg = False
    args.seg_visible = False
    
    args.useTex = False
    args.extractTex = False
    
    args.backbone = 'resnet50'
    
    return args


def main(input_path, output_dir=None, device='cuda'):
    input_path = Path(input_path)
    
    if output_dir is None:
        output_dir = input_path.parent / 'lmk'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    args = create_args(input_path, output_dir, device)
    
    recon_model = face_model(args)
    facebox_detector = face_box(args).detector
    
    if input_path.is_file():
        im_paths = [input_path]
    else:
        im_paths = get_data_path(str(input_path))
    
    if not im_paths:
        print(f"No images found in {input_path}")
        return
    
    print(f"Found {len(im_paths)} images for landmark extraction")
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for im_path in tqdm(im_paths, desc="Extracting landmarks"):
        img_name = Path(im_path).stem
        save_dir = output_dir / img_name
        
        if save_dir.exists() and (save_dir / 'landmarks.npy').exists():
            processed_count += 1
            continue
        
        try:
            im = Image.open(im_path).convert('RGB')
            trans_params, im_tensor = facebox_detector(im)
            
            if im_tensor is None:
                print(f"\nWarning: No face detected in {img_name}, skipping...")
                skipped_count += 1
                continue
            
            recon_model.input_img = im_tensor.to(args.device)
            results = recon_model.forward()
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            my_visualize = visualize(results, args)
            
            my_visualize.visualize_and_output(
                trans_params, 
                cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR),
                str(save_dir),
                'landmarks'
            )
            
            processed_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            error_count += 1
            continue
    
    print(f"\nLandmark extraction completed:")
    print(f"  - Processed: {processed_count} images")
    if skipped_count > 0:
        print(f"  - Skipped (no face): {skipped_count} images")
    if error_count > 0:
        print(f"  - Errors: {error_count} images")
    print(f"  - Results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA-V3 Inference for Landmark Extraction')
    
    parser.add_argument('-i', '--inputpath', required=True, type=str,
                        help='Path to the test data, should be an image file or folder')
    parser.add_argument('-o', '--output_dir', default=None, type=str,
                        help='Path to the output directory where results will be stored')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Set device, cuda or cpu')
    
    args = parser.parse_args()
    
    main(
        input_path=args.inputpath,
        output_dir=args.output_dir,
        device=args.device
    )
