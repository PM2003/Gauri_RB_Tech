"""Full preprocessing + inference pipeline for RB_Technology Virtual Try-On."""
import os
from PIL import Image


BASE       = '/content'
INPUTS     = f'{BASE}/inputs/test'
REPO       = f'{BASE}/RB_Technology'


def resize_images(folder: str, size=(768, 1024)) -> None:
    for fname in os.listdir(folder):
        if fname.startswith('.'):
            continue
        path = os.path.join(folder, fname)
        img  = Image.open(path).resize(size)
        img.save(path)
    print(f'[INFO] Resized all images in {folder}')


def write_pairs_file(pairs_path: str) -> None:
    models  = sorted(f for f in os.listdir(f'{INPUTS}/image') if not f.startswith('.'))
    clothes = sorted(f for f in os.listdir(f'{INPUTS}/cloth') if not f.startswith('.'))
    with open(pairs_path, 'w') as fp:
        for m, c in zip(models, clothes):
            fp.write(f'{m} {c}\n')
    print(f'[INFO] Wrote {len(models)} pairs to {pairs_path}')


def run_preprocessing() -> None:
    # 1. Resize cloth images
    resize_images(f'{INPUTS}/cloth')

    # 2. Generate cloth masks with U2NET
    os.chdir(REPO)
    os.system('rm -rf /content/inputs/test/cloth/.ipynb_checkpoints')
    os.system('python cloth_mask.py')

    # 3. Remove background from person images
    os.chdir(BASE)
    os.system('python /content/RB_Technology/remove_bg.py')

    # 4. Human parsing (Self-Correction Human Parsing)
    os.system(
        "python3 /content/Self-Correction-Human-Parsing/simple_extractor.py "
        "--dataset 'lip' "
        "--model-restore '/content/Self-Correction-Human-Parsing/checkpoints/final.pth' "
        "--input-dir '/content/inputs/test/image' "
        "--output-dir '/content/inputs/test/image-parse'"
    )

    # 5. Pose estimation (OpenPose)
    os.chdir(BASE)
    os.system(
        "cd openpose && "
        "./build/examples/openpose/openpose.bin "
        "--image_dir /content/inputs/test/image/ "
        "--write_json /content/inputs/test/openpose-json/ "
        "--display 0 --render_pose 0 --hand"
    )
    os.system(
        "cd openpose && "
        "./build/examples/openpose/openpose.bin "
        "--image_dir /content/inputs/test/image/ "
        "--write_images /content/inputs/test/openpose-img/ "
        "--display 0 --render_pose 1 --hand --disable_blending true"
    )


def run_inference() -> None:
    pairs_path = f'{BASE}/inputs/test_pairs.txt'
    write_pairs_file(pairs_path)
    os.system(
        f"python {REPO}/test.py "
        f"--name output "
        f"--dataset_dir {BASE}/inputs "
        f"--checkpoint_dir {REPO}/checkpoints "
        f"--save_dir {BASE}/"
    )
    os.system(f'rm -rf {BASE}/inputs')
    os.system(f'rm -rf {BASE}/output/.ipynb_checkpoints')


if __name__ == '__main__':
    run_preprocessing()
    run_inference()
