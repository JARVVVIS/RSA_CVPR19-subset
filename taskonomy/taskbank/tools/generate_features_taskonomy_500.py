###This is done for taskonomy new dataset for generating results for CVPR submission
###File to generate encoder features of all the tasks and final layers
### slight modification in for loop order to generate features faster
from __future__ import absolute_import, division, print_function

import argparse
import importlib
import itertools
import matplotlib
import glob
matplotlib.use('Agg')
import time
from   multiprocessing import Pool
import numpy as np
import os
import pdb
import pickle
import subprocess
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import threading
import scipy.misc
from skimage import color
import init_paths
from models.sample_models import *
from lib.data.synset import *
import scipy
import skimage
import skimage.io
import transforms3d
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from task_viz import *
import random
import utils
import models.architectures as architectures
from   data.load_ops import resize_rescale_image
from   data.load_ops import rescale_image
import utils
import lib.data.load_ops as load_ops
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Viz Single Task')

parser.add_argument('--task', dest='task')
parser.set_defaults(task='NONE')

parser.add_argument('--img', dest='im_name')
parser.set_defaults(im_name='NONE')

parser.add_argument('--save_dir', dest='save_dir')
parser.set_defaults(save_dir='./taskonomy_feats_taskonomy500')

parser.add_argument('--img_dir', dest='img_dir')
parser.set_defaults(img_dir='/home/kshitij/projects/taskonomy/deeplab_v3/taskonomy_500_images')

parser.add_argument('--store', dest='store_name')
parser.set_defaults(store_name='./rsa_test_ihlen_200_features_crap')

parser.add_argument('--store-rep', dest='store_rep', action='store_true')
parser.set_defaults(store_rep=False)

parser.add_argument('--store-pred', dest='store_pred', action='store_true')
parser.set_defaults(store_pred=True)

parser.add_argument('--on-screen', dest='on_screen', action='store_true')
parser.set_defaults(on_screen=False)

tf.logging.set_verbosity(tf.logging.ERROR)

list_of_tasks = 'autoencoder keypoint2d rgb2depth' 
list_of_tasks = list_of_tasks.split()
#list_of_tasks = list_of_tasks[-3:]
# tasks with fc layers in decoder
#list_of_tasks = ['class_places']
fc_task_list = 'class_1000 class_places vanishing_point jigsaw room_layout vanishing_point'
fc_task_list = fc_task_list.split()
#encoder_save_list = 'encoder_output'
#feedforward_encoder_save_list = 'encoder_output'
encoder_save_list = 'encoder/block1 encoder/block2 encoder/block3 encoder/block4 encoder_output'
feedforward_encoder_save_list = 'feedforward/encoder/block1 feedforward/encoder/block2 feedforward/encoder/block3 feedforward/encoder/block4 encoder_output'
feedforward_encoder_save_list = feedforward_encoder_save_list.split()
encoder_save_list = encoder_save_list.split()

def generate_cfg(task):
    repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    CONFIG_DIR = os.path.join(repo_dir, 'experiments/final', task)
    ############## Load Configs ##############
    import utils
    import data.load_ops as load_ops
    from   general_utils import RuntimeDeterminedEnviromentVars
    cfg = utils.load_config( CONFIG_DIR, nopause=True )
    RuntimeDeterminedEnviromentVars.register_dict( cfg )
    cfg['batch_size'] = 1
    if 'batch_size' in cfg['encoder_kwargs']:
        cfg['encoder_kwargs']['batch_size'] = 1
    cfg['model_path'] = os.path.join( repo_dir, 'temp', task, 'model.permanent-ckpt' )
    cfg['root_dir'] = repo_dir
    return cfg

def run_to_task():
    import general_utils
    from   general_utils import RuntimeDeterminedEnviromentVars
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    tf.logging.set_verbosity(tf.logging.ERROR)
    image_list = glob.glob(args.img_dir +"/*.png")
    image_list.sort()
    #image_list=image_list[:5]
    print(len(image_list))
    for task in tqdm(list_of_tasks):
        print("Task is ", task)
        if task not in list_of_tasks:
            raise ValueError('Task not supported')
        low_sat_tasks = 'autoencoder curvature denoise edge2d edge3d \
        keypoint2d keypoint3d \
        reshade rgb2depth rgb2mist rgb2sfnorm \
        segment25d segment2d room_layout'.split()
        cfg = generate_cfg(task)
        if task in low_sat_tasks:
            cfg['input_preprocessing_fn'] = load_ops.resize_rescale_image_low_sat
        print("Doing {task}".format(task=task))
        general_utils = importlib.reload(general_utils)
        tf.reset_default_graph()
        training_runners = { 'sess': tf.InteractiveSession(), 'coord': tf.train.Coordinator() }

        ############## Set Up Inputs ##############
        # tf.logging.set_verbosity( tf.logging.INFO )
        setup_input_fn = utils.setup_input
        inputs = setup_input_fn( cfg, is_training=False, use_filename_queue=False )
        RuntimeDeterminedEnviromentVars.load_dynamic_variables( inputs, cfg )
        RuntimeDeterminedEnviromentVars.populate_registered_variables()
        start_time = time.time()

        ############## Set Up Model ##############
        model = utils.setup_model( inputs, cfg, is_training=False )
        m = model[ 'model' ]
        model[ 'saver_op' ].restore( training_runners[ 'sess' ], cfg[ 'model_path' ] )
        #Prints all the class variables and functions
        print(dir(m))
        for img_name in tqdm(image_list):
            filename = img_name.split('/')[-1].split('.')[0]

            img = load_raw_image_center_crop( img_name )
            img = skimage.img_as_float(img)
            scipy.misc.toimage(np.squeeze(img), cmin=0.0, cmax=1.0).save(img_name)

            # Since we observe that areas with pixel values closes to either 0 or 1 sometimes overflows, we clip pixels value



            if task == 'jigsaw' :
                img = cfg[ 'input_preprocessing_fn' ]( img, target=cfg['target_dict'][random.randint(0,99)],
                                                        **cfg['input_preprocessing_fn_kwargs'] )
            else:
                img = cfg[ 'input_preprocessing_fn' ]( img, **cfg['input_preprocessing_fn_kwargs'] )

            img = img[np.newaxis,:]




            if task in fc_task_list:
                predicted, representation, decoder_features, encoder_features = training_runners['sess'].run(
                        [ m.decoder_output,  m.encoder_output, m.metric_endpoints, m.encoder_endpoints ], feed_dict={m.input_images: img} )
            else:
                predicted, representation, decoder_features, encoder_features = training_runners['sess'].run(
                        [ m.decoder_output,  m.encoder_output, m.decoder_endpoints, m.encoder_endpoints ], feed_dict={m.input_images: img} )
            #np.save(save_path,value)
            #for name,value in decoder_features.items():
            #    print (name)
            ## CKD : Uncomment below for loop
            for name,value in encoder_features.items():
                if name in encoder_save_list or name in feedforward_encoder_save_list:
                    #print (name)
                    name = name.replace('/', '_')
                    save_path = os.path.join(args.save_dir,filename+"_"+task + "_" + name + ".npy")
                    np.save(save_path,value)
            if args.store_rep:
                s_name, file_extension = os.path.splitext(args.store_name)
                with open('{}.npy'.format(s_name), 'wb') as fp:
                    np.save(fp, np.squeeze(representation))

            if args.store_pred:
                save_path = os.path.join(args.save_dir,filename+"_"+task + "_" + "prediction" + ".npy")
                with open(save_path, 'wb') as fp:
                    np.save(fp, np.squeeze(predicted))
            #if task == 'segment2d' or task == 'segment25d':
            #    segmentation_pca(predicted, args.store_name)
                #return
            #if task == 'colorization':
            #    single_img_colorize(predicted, img , args.store_name)
                #return

            #if task == 'curvature':
            #    curvature_single_image(predicted, args.store_name)
                #return

            #just_rescale = ['autoencoder', 'denoise', 'edge2d',
            #                'edge3d', 'keypoint2d', 'keypoint3d',
            #                'reshade', 'rgb2sfnorm' ]

            #if task in just_rescale:
            #    print(args.store_name)
                #simple_rescale_img(predicted, args.store_name)
                #return

            #just_clip = ['rgb2depth', 'rgb2mist']
            #if task in just_clip:
            #    depth_single_image(predicted, args.store_name)
            #    #return

            #if task == 'inpainting_whole':
            #    inpainting_bbox(predicted, args.store_name)
                #return

            #if task == 'segmentsemantic':
            #    semseg_single_image( predicted, img, args.store_name)
                #return

            #if task in ['class_1000', 'class_places']:
             #   print("The shape of predicted is ---------------", predicted.shape)
                #classification(predicted, synset, args.store_name)
                #return

            #if task == 'vanishing_point':
            #    _ = plot_vanishing_point_smoothed(np.squeeze(predicted), (np.squeeze(img) + 1. )/2., args.store_name, [])
                #return

            #if task == 'room_layout':
            #    mean = np.array([0.006072743318127848, 0.010272365569691076, -3.135909774145468,
            #                    1.5603802322235532, 5.6228218371102496e-05, -1.5669352793761442,
            #                                5.622875878174759, 4.082800262277375, 2.7713941642895956])
            #    std = np.array([0.8669452525283652, 0.687915294956501, 2.080513632043758,
            #                    0.19627420479282623, 0.014680602791251812, 0.4183827359302299,
            #                                3.991778013006544, 2.703495278378409, 1.2269185938626304])
            #    predicted = predicted * std + mean
            #    plot_room_layout(np.squeeze(predicted), (np.squeeze(img) + 1. )/2., args.store_name, [], cube_only=True)
                #return

            #if task == 'jigsaw':
            #    predicted = np.argmax(predicted, axis=1)
            #    perm = cfg[ 'target_dict' ][ predicted[0] ]
            #    show_jigsaw((np.squeeze(img) + 1. )/2., perm, args.store_name)
                #return

            ############## Clean Up ##############
        training_runners[ 'coord' ].request_stop()
        training_runners[ 'coord' ].join()
        print("Done: {}".format(task))

        ############## Reset graph and paths ##############
        tf.reset_default_graph()
        training_runners['sess'].close()
    return

if __name__ == '__main__':
    run_to_task()
