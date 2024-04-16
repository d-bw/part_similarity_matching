# part_similarity_matching
extract the part features on the image and to match the drx/svg file 

# install the dependencies
pip install -r requirements.txt

# visualize the features on the image 
python ./utils/feature_extract.py -model_id facebook/sam-vit-base -image_path test_resource/test2.jpg
