#!/usr/bin/env python
import os
import numpy as np 
import tensorflow as tf
import argparse
from PIL import Image	# required to read indexed PNG files used in VOC labels
from glob import glob

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def main(args):
	imagesets_file = os.path.join(args.dataset_path, 'ImageSets',args.year,'train.txt')
	image_paths = []
	annotation_paths = []
	with open(imagesets_file,'r') as f:
		for folder in f:
			images = sorted(glob(os.path.join(args.dataset_path,'JPEGImages', args.res,folder.strip(),'*.jpg')))
			annotations = sorted(glob(os.path.join(args.dataset_path,'Annotations', args.res,folder.strip(),'*.png')))
			if len(images) != len(annotations):
				print("ERROR: Length of images and annotations don't match!")
				exit(1)
			image_paths.extend(images)
			annotation_paths.extend(annotations)

	tfrecords_filename = os.path.join(args.output_path, 'DAVIS_'+args.res+'.tfrecords')
	writer = tf.python_io.TFRecordWriter(tfrecords_filename)

	for img_path, annotation_path in zip(image_paths, annotation_paths):
		
		img = np.array(Image.open(img_path))
		annotation = Image.open(annotation_path)
		if annotation.mode is 'LA':
			annotation = annotation.convert('L')
		annotation = np.array(annotation)
		height = img.shape[0]
		width = img.shape[1]

		img_raw = img.tostring()
		annotation_raw = annotation.tostring()

		if len(annotation_raw) != 409920:
			print("Skipping ", img_path)
			continue

		example = tf.train.Example(features=tf.train.Features(feature={
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'image_raw': _bytes_feature(img_raw),
			'mask_raw': _bytes_feature(annotation_raw)}))
		writer.write(example.SerializeToString())
		if len(annotation_raw) != height*width:
			print('ERROR')
			exit(1)
		# if height*width != 409920:
		print("Processed ",img_path, annotation.shape, img.shape, len(annotation_raw), len(img_raw))
	writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a TFRecord file for DAVIS dataset')
    parser.add_argument('--dataset-path', dest='dataset_path', help='path to dataset directory', default='./DAVIS', type=str)
    parser.add_argument('--output-path', dest='output_path', help='path to output directory', default='./data', type=str)
    parser.add_argument('--resolution', dest='res', help='480p or 1080p', default='480p', type=str)
    parser.add_argument('--year', dest='year', help='2016 or 2017', default='2017', type=str)
    args = parser.parse_args()
    main(args)