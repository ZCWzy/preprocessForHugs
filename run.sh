echo ========================================
echo 1: Extract frames
echo ========================================
echo ========================================
echo 2: Masks
echo ========================================
cd /root/detectron2/demo
cd /root/preprocessForHugs
echo ========================================
echo 3: Sparse scene reconstrution
echo ========================================
cd /root/autodl-tmp/data/chaodupp
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
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --input /root/autodl-tmp/data/chaodupp/output/images/*.png --output /root/autodl-tmp/data/chaodupp/output/segmentations  --opts MODEL.WEIGHTS ./model_final_2d9806.pkl
conda deactivate
cd /root/preprocessForHugs
echo ========================================
echo 4: SMPL parameters
echo ========================================
cd /root/ROMP
conda activate ROMP
python -m romp.predict.image --inputs /root/autodl-tmp/data/chaodupp/output/images --output_dir /root/autodl-tmp/data/chaodupp/output/smpl_pred
conda deactivate
cd /root/preprocessForHugs
echo ========================================
echo 5: Image Conversion
echo ========================================
cd /root/preprocessForHugs
conda activate preprocessForHugs
mkdir /root/autodl-tmp/data/chaodupp/output/4D_humans/phalp
mkdir /root/autodl-tmp/data/chaodupp/output/4D_humans/sam_segmentations
python image_conversion.py --images_dir /root/autodl-tmp/data/chaodupp/output/images --seg_dir/root/autodl-tmp/data/chaodupp/output/segmentations --seg_output_dir /root/autodl-tmp/data/chaodupp/output/4D_humans/sam_segmentations --phalp_output_dir /root/autodl-tmp/data/chaodupp/output/4D_humans/phalp 
conda deactivate
cd /root/preprocessForHugs
echo ========================================
echo 7: Solve scale ambiguity
echo ========================================
cd /root/preprocessForHugs
conda activate preprocessForHugs
python export_alignment_myself.py --scene_dir /root/autodl-tmp/data/chaodupp/output/sparse --images_dir /root/autodl-tmp/data/chaodupp/output/images --raw_smpl /root/autodl-tmp/data/chaodupp/4d-humans/track_results.pkl 
conda deactivate
cd /root/preprocessForHugs