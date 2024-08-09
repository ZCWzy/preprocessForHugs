import argparse
import os
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, type=str, help='the path to the source video')
    opt = parser.parse_args()
    assert os.path.isfile(opt.video)

    steps = 10
    video_name = os.path.splitext(os.path.basename(opt.video))[0]
    code_dir = os.path.dirname(os.path.realpath(__file__))
    video_dir = os.path.dirname(os.path.abspath(opt.video))
    print(code_dir)


    commands = []
    commands.append(f'echo ========================================')
    commands.append(f'echo 1: Extract frames')
    commands.append(f'echo ========================================')
    if not os.path.isdir(os.path.join(video_dir, f'{video_name}/raw_720p')):
        commands.append(f'cd {code_dir}')
        commands.append('conda activate preprocessForHugs')

        capture = cv2.VideoCapture(opt.video)
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        #根据自己视频可以微调
        commands.append(f'python save_video_frames.py --video {opt.video} --save_to {os.path.join(video_dir, video_name, "raw_720p")}  --width {frame_width} --height {frame_height} --every 10 --skip=0')
        commands.append('conda deactivate')


    # Generate masks 
    commands.append(f'echo ========================================')
    commands.append(f'echo 2: Masks')
    commands.append(f'echo ========================================')
    commands.append(f'cd /root/detectron2/demo')
    if not os.path.isfile(os.path.join('/root/detectron2/demo/model_final_2d9806.pkl')):
        commands.append('wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl')
    if not os.path.isdir(os.path.join(video_dir, f'{video_name}/raw_masks')):
        commands.append('conda activate ROMP')
        commands.append(f'python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input {os.path.join(video_dir, f"{video_name}/raw_720p/*.png")} --output {os.path.join(video_dir, f"{video_name}/raw_masks")}  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl')

        commands.append(f'cd {video_dir}/{video_name}/raw_masks')
        commands.append('rm -rf .ipynb_checkpoints') #jupyterlab会出现的文件夹
        commands.append('rm *.png.png')
        commands.append('rm -rf det')
        commands.append(f'cd {video_dir}/{video_name}/raw_720p')
        commands.append(f'rm -rf .ipynb_checkpoints')
    commands.append(f'cd {code_dir}')
    # 可能会因为pillow版本问题，出现 AttributeError: module 'PIL.Image' has no attribute 'LINEAR' 问题。
    # 解决办法：依据 https://github.com/facebookresearch/detectron2/issues/5010#issuecomment-1629504706
    # 将报错位置换成 PIL.Image.BILINEAR 即可

    #注意，如果你换了显卡，可能需要重新编译安装colmap才能进行exhaustive_matcher 
    commands.append(f'echo ========================================')
    commands.append(f'echo 3: Sparse scene reconstrution')
    commands.append(f'echo ========================================')
    commands.append(f'cd {os.path.join(video_dir, video_name)}')
    #colmap
    if not os.path.isdir(os.path.join(video_dir, video_name, 'output/sparse')):
        commands.append('mkdir recon')
        commands.append('colmap feature_extractor --database_path ./recon/db.db --image_path ./raw_720p --ImageReader.mask_path ./raw_masks --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pool=true --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1')
        commands.append('colmap exhaustive_matcher --database_path ./recon/db.db --SiftMatching.guided_matching=true')
        commands.append('mkdir -p ./recon/sparse')
        commands.append('colmap mapper --database_path ./recon/db.db --image_path ./raw_720p --output_path ./recon/sparse')
        commands.append('if [ -d "./recon/sparse/1" ]; then echo "Bad reconstruction"; exit 1; else echo "Ok"; fi')
        commands.append('mkdir -p ./recon/dense')
        commands.append('colmap image_undistorter --image_path raw_720p --input_path ./recon/sparse/0/ --output_path ./recon/dense')
        commands.append('colmap patch_match_stereo --workspace_path ./recon/dense')
        commands.append('colmap model_converter --input_path ./recon/dense/sparse/ --output_path ./recon/dense/sparse --output_type=TXT')
        commands.append('mkdir ./output')
        commands.append('cp -r ./recon/dense/images ./output/images')
        commands.append('cp -r ./recon/dense/stereo/depth_maps ./output/depth_maps')
        commands.append('cp -r ./recon/dense/sparse ./output/sparse')
        commands.append('cd ./output/images')
        commands.append('rm -rf .ipynb_checkpoints')
    commands.append(f'cd {code_dir}')


    #这一步不能少！
    commands.append(f'echo ========================================')
    commands.append(f'echo 4/: Masks for rectified images')
    commands.append(f'echo ========================================')
    commands.append(f'cd /root/detectron2/demo')
    if not os.path.isdir(os.path.join(video_dir, f'{video_name}/output/segmentations')):
        commands.append('conda activate ROMP')
        commands.append(f'python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input {os.path.join(video_dir, f"{video_name}/output/images/*.png")} --output {os.path.join(video_dir, f"{video_name}/output/segmentations")}  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl')
        commands.append('conda deactivate')
        commands.append(f'cd {os.path.join(video_dir, f"{video_name}/output/segmentations")}')
        commands.append('rm -rf .ipynb_checkpoints') #jupyterlab会出现的文件夹
        commands.append('rm *.png.png')
        commands.append('rm -rf det')
    commands.append(f'cd {code_dir}')

    # 移除了densepose和使用ROMP预测人体姿势的代码

    #mask反色和png to jpg
    commands.append(f'echo ========================================')
    commands.append(f'echo 5: Image Conversion')
    commands.append(f'echo ========================================')
    commands.append(f'cd {code_dir}')
    if not os.path.isfile(os.path.join(video_dir, f'{video_name}/output/alignments.npy')):
        commands.append('conda activate preprocessForHugs')
        commands.append('mkdir '+os.path.join(video_dir,f"{video_name}/output/4d_humans/"))
        if not os.path.exists(os.path.join(video_dir,f"{video_name}/output/4d_humans/phalp")):
            commands.append("mkdir "+os.path.join(video_dir,f"{video_name}/output/4d_humans/phalp"))
        if not os.path.exists(os.path.join(video_dir,f"{video_name}/output/4d_humans/sam_segmentations")):
            commands.append("mkdir "+os.path.join(video_dir,f"{video_name}/output/4d_humans/sam_segmentations"))
        commands.append(f'python image_conversion.py --images_dir {os.path.join(video_dir, f"{video_name}/output/images")} --seg_dir {os.path.join(video_dir, f"{video_name}/output/segmentations")} --seg_output_dir {os.path.join(video_dir, f"{video_name}/output/4d_humans/sam_segmentations")} --phalp_output_dir {os.path.join(video_dir, f"{video_name}/output/4d_humans/phalp")} ')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')

    # SMPL parameters estimation
    commands.append(f'echo ========================================')
    commands.append(f'echo 6: SMPL parameters')
    commands.append(f'echo ========================================')
    commands.append('cd /root/4D-Humans')

    commands.append(f'conda activate 4D-humans')
    commands.append(f'python track.py video.source={os.path.join(video_dir, f"{video_name}/output/4d_humans/phalp")} ')
    commands.append(f'python demo.py --img_folder {os.path.join(video_dir, f"{video_name}/output/images")} \
    --out_folder ./demo_out --batch_size=48')
    commands.append(f'cp ./outputs/results/track_result.pkl {os.path.join(video_dir, f"{video_name}/output/4d_humans/track_results.pkl")}')
    commands.append(f'conda deactivate')
    
    # Solve scale ambiguity and make smpl_optimized_aligned_scale.npz
    commands.append(f'echo ========================================')
    commands.append(f'echo 7: Solve scale ambiguity')
    commands.append(f'echo ========================================')
    commands.append(f'cd {code_dir}')
    if not os.path.isfile(os.path.join(video_dir, f'{video_name}/output/alignments.npy')):
        commands.append('conda activate preprocessForHugs')
        commands.append(f'python export_alignment_myself.py --scene_dir {os.path.join(video_dir, f"{video_name}/output/sparse")} --images_dir {os.path.join(video_dir, f"{video_name}/output/images")} --raw_smpl {os.path.join(video_dir, f"{video_name}/output/4d_humans/track_results.pkl")} ')
        commands.append(f'mv ./smpl_optimized_aligned_scale.npz {os.path.join(video_dir, f"{video_name}/output/4d_humans/")}')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')

    commands.append(f'echo ========================================')
    commands.append(f'echo 8: make dense pose')
    commands.append(f'echo ========================================')
    commands.append(f'cd {code_dir}')
    if not os.path.isfile(os.path.join(video_dir, f'{video_name}/output/densepose/')):
        commands.append(f'mkdir {os.path.join(video_dir, f"{video_name}/output/densepose")}')
        commands.append('conda activate preprocessForHugs')
        commands.append(f'python make_densepose.py --input {code_dir}/dp_00000.png.npy --images_dir {os.path.join(video_dir, f"{video_name}/output/images")} --output_dir {os.path.join(video_dir, f"{video_name}/output/densepose")}')
        commands.append('conda deactivate')
    commands.append(f'cd {code_dir}')


    print(*commands, sep='\n')
    with open("run.sh", "w") as outfile:
        outfile.write("\n".join(commands))

if __name__ == "__main__":
    main()