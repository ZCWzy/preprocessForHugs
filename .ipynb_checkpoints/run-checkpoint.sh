echo ========================================
echo 1/10: Extract frames
echo ========================================
conda activate preprocessForHugs
python save_video_frames.py --video E:\practice\preprocessForHugs\myself.mp4 --save_to E:\practice\preprocessForHugs\myself\raw_720p  --width 1280 --height 720 --every 10 --skip=0
conda deactivate
echo ========================================
echo 2/10: Masks
echo ========================================
cd E:\practice\preprocessForHugs\detectron2/demo
conda activate preprocessForHugs
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input E:\practice\preprocessForHugs\myself/raw_720p/*.png --output E:\practice\preprocessForHugs\myself/raw_masks  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl MODEL.DEVICE cpu
conda deactivate
cd E:\practice\preprocessForHugs
echo ========================================
echo 3/10: Sparse scene reconstrution
echo ========================================
cd E:\practice\preprocessForHugs\myself
mkdir recon
D:/code/preprocessForHugs/COLMAP-3.6-windows-no-cuda/colmap.bat feature_extractor --database_path ./recon/db.db --image_path ./raw_720p --ImageReader.mask_path ./raw_masks --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pool=true --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1
D:/code/preprocessForHugs/COLMAP-3.6-windows-no-cuda/colmap.bat exhaustive_matcher --database_path ./recon/db.db --SiftMatching.guided_matching=true
mkdir -p ./recon/sparse
D:/code/preprocessForHugs/COLMAP-3.6-windows-no-cuda/colmap.bat mapper --database_path ./recon/db.db --image_path ./raw_720p --output_path ./recon/sparse
if [ -d "./recon/sparse/1" ]; then echo "Bad reconstruction"; exit 1; else echo "Ok"; fi
mkdir -p ./recon/dense
D:/code/preprocessForHugs/COLMAP-3.6-windows-no-cuda/colmap.bat image_undistorter --image_path raw_720p --input_path ./recon/sparse/0/ --output_path ./recon/dense
D:/code/preprocessForHugs/COLMAP-3.6-windows-no-cuda/colmap.bat patch_match_stereo --workspace_path ./recon/dense
D:/code/preprocessForHugs/COLMAP-3.6-windows-no-cuda/colmap.bat model_converter --input_path ./recon/dense/sparse/ --output_path ./recon/dense/sparse --output_type=TXT
mkdir ./output
cp -r ./recon/dense/images ./output/images
cp -r ./recon/dense/stereo/depth_maps ./output/depth_maps
cp -r ./recon/dense/sparse ./output/sparse
cd E:\practice\preprocessForHugs
echo ========================================
echo 4/10: SMPL parameters
echo ========================================
cd E:\practice\preprocessForHugs\ROMP
wget https://github.com/jiangwei221/ROMP/releases/download/v1.1/model_data.zip
unzip model_data.zip
wget https://github.com/Arthur151/ROMP/releases/download/v1.1/trained_models_try.zip
unzip trained_models_try.zip
conda activate ROMP
python -m romp.predict.image --inputs E:\practice\preprocessForHugs\myself/output/images --output_dir E:\practice\preprocessForHugs\myself/output/smpl_pred
conda deactivate
cd E:\practice\preprocessForHugs
echo ========================================
echo 5/10: Solve scale ambiguity
echo ========================================
cd E:\practice\preprocessForHugs
conda activate preprocessForHugs
python export_alignment.py --scene_dir E:\practice\preprocessForHugs\myself/output/sparse --images_dir E:\practice\preprocessForHugs\myself/output/images --raw_smpl E:\practice\preprocessForHugs\myself/4d-humans/track_results.pkl 
conda deactivate
cd E:\practice\preprocessForHugs