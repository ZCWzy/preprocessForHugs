echo ========================================
echo 1: Extract frames
echo ========================================
conda activate preprocessForHugs
python save_video_frames.py --video /root/autodl-tmp/data/myself.mp4 --save_to /root/autodl-tmp/data/myself/raw_720p  --width 544 --height 960 --every 10 --skip=0
conda deactivate
echo ========================================
echo 2: Masks
echo ========================================
cd /root/detectron2/demo
conda activate ROMP
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /root/autodl-tmp/data/myself/raw_720p/*.png --output /root/autodl-tmp/data/myself/raw_masks  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl
cd /root/autodl-tmp/data/myself/raw_masks
rm -rf .ipynb_checkpoints
rm *.png.png
rm -rf det
cd /root/autodl-tmp/data/myself/raw_720p
rm -rf .ipynb_checkpoints
cd /root/preprocessForHugs
echo ========================================
echo 3: Sparse scene reconstrution
echo ========================================
cd /root/autodl-tmp/data/myself
mkdir recon
colmap feature_extractor --database_path ./recon/db.db --image_path ./raw_720p --ImageReader.mask_path ./raw_masks --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pool=true --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1
colmap exhaustive_matcher --database_path ./recon/db.db --SiftMatching.guided_matching=true
mkdir -p ./recon/sparse
colmap mapper --database_path ./recon/db.db --image_path ./raw_720p --output_path ./recon/sparse
if [ -d "./recon/sparse/1" ]; then echo "Bad reconstruction"; exit 1; else echo "Ok"; fi
mkdir -p ./recon/dense
colmap image_undistorter --image_path raw_720p --input_path ./recon/sparse/0/ --output_path ./recon/dense
colmap patch_match_stereo --workspace_path ./recon/dense
colmap model_converter --input_path ./recon/dense/sparse/ --output_path ./recon/dense/sparse --output_type=TXT
mkdir ./output
cp -r ./recon/dense/images ./output/images
cp -r ./recon/dense/stereo/depth_maps ./output/depth_maps
cp -r ./recon/dense/sparse ./output/sparse
cd /root/preprocessForHugs
echo ========================================
echo 4/: Masks for rectified images
echo ========================================
cd /root/detectron2/demo
conda activate ROMP
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /root/autodl-tmp/data/myself/output/images/*.png --output /root/autodl-tmp/data/myself/output/segmentations  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl
conda deactivate
cd /root/autodl-tmp/data/myself/output/segmentations
rm -rf .ipynb_checkpoints
rm *.png.png
rm -rf det
cd /root/preprocessForHugs
echo ========================================
echo 5: Image Conversion
echo ========================================
cd /root/preprocessForHugs
conda activate preprocessForHugs
mkdir /root/autodl-tmp/data/myself/output/4d_humans/phalp
mkdir /root/autodl-tmp/data/myself/output/4d_humans/sam_segmentations
python image_conversion.py --images_dir /root/autodl-tmp/data/myself/output/images --seg_dir/root/autodl-tmp/data/myself/output/segmentations --seg_output_dir /root/autodl-tmp/data/myself/output/4D_humans/sam_segmentations --phalp_output_dir /root/autodl-tmp/data/myself/output/4D_humans/phalp 
conda deactivate
cd /root/preprocessForHugs
echo ========================================
echo 6: SMPL parameters
echo ========================================
cd /root/4D-humans
conda activate 4D-humans
python track.py video.source=/root/autodl-tmp/data/myself/output/4d_humans/phalp 
python demo.py --img_folder /root/autodl-tmp/data/myself/output/images     --out_folder ./demo_out --batch_size=48
cp /output/results/track_results.pkl /root/autodl-tmp/data/myself/4d-humans/track_results.pkl
conda deactivate
echo ========================================
echo 7: Solve scale ambiguity
echo ========================================
cd /root/preprocessForHugs
conda activate preprocessForHugs
python export_alignment_myself.py --scene_dir /root/autodl-tmp/data/myself/output/sparse --images_dir /root/autodl-tmp/data/myself/output/images --raw_smpl /root/autodl-tmp/data/myself/4d-humans/track_results.pkl 
conda deactivate
cd /root/preprocessForHugs
echo ========================================
echo 8: make dense pose
echo ========================================
cd /root/preprocessForHugs
mkdir /root/autodl-tmp/data/myself/output/densepose
conda activate preprocessForHugs
python make_densepose.py --input /root/preprocessForHugs/dp_00000.png.npy --images_dir /root/autodl-tmp/data/myself/output/images --output_dir /root/autodl-tmp/data/myself/output/densepose
conda deactivate
cd /root/preprocessForHugs