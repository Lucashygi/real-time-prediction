import os
import cv2
import warnings
import matplotlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def rescale_images(percentsizeimage,image):
	#for root, dirs, files in os.walk(directory):
	#for filename in files:
	scale_percent = percentsizeimage /100
	width = int(image.shape[1] * scale_percent)
	height = int(image.shape[0] * scale_percent)
	dsize = (width, height)
	output = cv2.resize(image, dsize,cv2.INTER_NEAREST)
	return output

matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')

from models.research.object_detection.utils import label_map_util, config_util
from models.research.object_detection.utils import visualization_utils as viz_utils
from models.research.object_detection.builders import model_builder


#recover our saved model
def load_image_into_numpy_array(image):
	return image.astype(np.uint8)

def get_model_detection_function(model):
	"""Get a tf.function for detection."""

	@tf.function
	def detect_fn(image):
		"""Detect objects in image."""

		image, shapes = model.preprocess(image)
		prediction_dict = model.predict(image, shapes)
		detections = model.postprocess(prediction_dict, shapes)

		return detections, prediction_dict, tf.reshape(shapes, [-1])
	
	return detect_fn

pipeline_config = './models/efficientdet/pipeline_file.config'
label_map_path  = './models/efficientdet/label_map.pbtxt'
model_dir 			= './models/efficientdet/ckpt-2'
checkpoint 			= './models/efficientdet/ckpt-2'
inputVideo  		= './videos/input.mp4' # or 0 to started Webcan
outputVideo 		= './output.mp4'

configs 				= config_util.get_configs_from_pipeline_file(pipeline_config)

#generally you want to put the last ckpt from training in here
model_config = configs['model']
detection_model = model_builder.build(
	model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
			model=detection_model)
ckpt.restore(os.path.join(checkpoint))

detect_fn = get_model_detection_function(detection_model)


#map labels for inference decoding
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
		label_map,
		max_num_classes=label_map_util.get_max_label_map_index(label_map),
		use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

cap = cv2.VideoCapture(inputVideo)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('./outputff.mp4',fourcc, 20.0, (640,480))
out = cv2.VideoWriter(outputVideo,cv2.VideoWriter_fourcc(*'mp4v'), 20, 
	(frame_width,frame_height)
)

while(True):
		# Capture frame-by-frame
		
		ret,image_np = cap.read()
		
		if image_np is None:
			break

		image_np = load_image_into_numpy_array(image_np)

		input_tensor = tf.convert_to_tensor(
		np.expand_dims(image_np, 0), dtype=tf.float32)
		
		detections, predictions_dict, shapes = detect_fn(input_tensor)

		label_id_offset = 1
		image_np_with_detections = image_np.copy()

		viz_utils.visualize_boxes_and_labels_on_image_array(
					image_np_with_detections,
					detections['detection_boxes'][0].numpy(),
					(detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
					detections['detection_scores'][0].numpy(),
					category_index,
					use_normalized_coordinates=True,
					max_boxes_to_draw=200,
					min_score_thresh=.75,
					agnostic_mode=False,
		)

		im_rgb = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
		# Display the resulting frame
		out.write(image_np_with_detections)
		cv2.imshow('frame',image_np_with_detections)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
				break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()