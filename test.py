#!/usr/bin/env python3
import os, sys, traceback, subprocess, shutil, argparse
import numpy as np
from imageio import imread
from skimage import color

def read_image(filename, representation):
    im = imread(filename)
    if representation == 1 and im.ndim == 3 and im.shape[2] == 3:
        im = color.rgb2gray(im)
    if im.dtype == np.uint8:
        im = im.astype(np.float64) / 255.0
    if im.dtype != np.float64:
        im = im.astype(np.float64)
    return im

def presubmit():
    print ('Ex3 Presubmission Script\n========================\n')
    disclaimer="""
    Disclaimer
    ----------
    The purpose of this script is to make sure that your code is compliant
    with the exercise API and some of the requirements
    The script does not test the quality of your results.
    Don't assume that passing this script will guarantee that you will get
    a high grade in the exercise
    """
    # print (disclaimer)
    
    # print('=== Check Submission ===\n')
    # if not os.path.exists('current/README.md'):
    #     print ('No readme!')
    #     return False
    # else:
    #     print('README file:\n')
    #     with open ('current/README.md') as f:
    #         print(f.read())
    
    #print('\n=== Answers to questions ===')
    #for q in [1,2,3]:
    #    if not os.path.exists('current/answer_q%d.txt'%q):
    #        print ('No answer_q%d.txt!'%q)
    #        return False
    #    print ('\nAnswer to Q%d:'%q)
    #    with open('current/answer_q%d.txt'%q) as f:
    #        print (f.read())
    print('=== Load Student Library ===\n')
    print('Loading...')
    sys.stdout.flush()
    sol3 = None
    try:
        import sol3 as sol3
    except Exception:
        print(traceback.format_exc())
        print('Unable to import the solution.')
        return False
    print ('\n=== Section 3.1 ===\n')
    im_orig = read_image('monkey.jpg', 1)
    try:
        print ('Trying to build Gaussian pyramid...')
        sys.stdout.flush()
        gpyr, filter_vec = sol3.build_gaussian_pyramid(im_orig, 3, 3)
        print ('\tPassed!')
        print ('Checking Gaussian pyramid type and structure...')
        if type(gpyr) is not list:
            raise ValueError('Returned pyramid is not a list type. It is %s instead.' % str(type(gpyr)))
        if len(gpyr) != 3:
            raise ValueError('Length of pyramid is wrong. Expecting length 3 list.')
        if type(filter_vec) != np.ndarray:
            raise ValueError('filter_vec is not a numpy array, but %s' % type(filter_vec))
        if filter_vec.shape != (1, 3):
            raise ValueError('Wrong blur filter size. Expecting 1x3')
        if any([l.dtype != np.float64 for l in gpyr]):
            raise ValueError('At least one of the levels in the pyramid is not float64.')
        print ('\tPassed!')
    except Exception:
        print(traceback.format_exc())
        return False
    try:
        print ('Trying to build Laplacian pyramid...')
        sys.stdout.flush()
        lpyr, filter_vec = sol3.build_laplacian_pyramid(im_orig, 3, 3)
        print ('\tPassed!')
        print ('Checking Laplacian pyramid type and structure...')
        if type(lpyr) is not list:
            raise ValueError('Returned pyramid is not a list type. It is %s instead.' % str(type(lpyr)))
        if len(lpyr) != 3:
             raise ValueError('Length of pyramid is wrong. Expecting length 3 list.')
        if type(filter_vec) != np.ndarray:
            raise ValueError('filter_vec is not a numpy array, but %s.' % type(filter_vec))
        if filter_vec.shape != (1, 3):
            raise ValueError('Wrong blur filter size. Expecting 1x3')
        if any([l.dtype != np.float64 for l in lpyr]):
            raise ValueError('At least one of the levels in the pyramid is not float64.')
        print ('\tPassed!')
    except Exception:
        print(traceback.format_exc())
        return False

    print ('\n=== Section 3.2 ===\n')
    try:
        print ('Trying to build Laplacian pyramid...')
        sys.stdout.flush()
        lpyr, filter_vec = sol3.build_laplacian_pyramid(im_orig, 3, 3)
        print ('\tPassed!')
        print ('Trying to reconstruct image from pyramid... (we are not checking for quality!)')
        sys.stdout.flush()
        im_r = sol3.laplacian_to_image(lpyr, filter_vec, [1, 1, 1])
        print ('\tPassed!')
        print ('Checking reconstructed image type and structure...')
        if im_r.dtype != np.float64:
            raise ValueError('Reconstructed image is not float64. It is %s instead.' % str(im_r.dtype))
        if im_orig.shape != im_r.shape:
            raise ValueError('Reconstructed image is not the same size as the original image.')
        print ('\tPassed!')
    except Exception:
        print(traceback.format_exc())
        return False

    print ('\n=== Section 3.3 ===\n')
    try:
        print ('Trying to build Gaussian pyramid...')
        sys.stdout.flush()
        gpyr, filter_vec = sol3.build_gaussian_pyramid(im_orig, 3, 3)
        print ('\tPassed!')
        print ('Trying to render pyramid to image...')
        sys.stdout.flush()
        im_pyr = sol3.render_pyramid(gpyr, 2)
        print ('\tPassed!')
        print ('Checking structure of returned image...')
        if im_pyr.shape != (400, 600):
            raise ValueError('Rendered pyramid is not the expected size. Expecting 400x600. Found %s.' % str(im_pyr.shape))
        print ('\tPassed!')
        print ('Trying to display image... (if DISPLAY env var not set, assumes running w/o screen)')
        sys.stdout.flush()
        sol3.display_pyramid(gpyr, 2)
        print ('\tPassed!')
    except Exception:
        print(traceback.format_exc())
        return False

    print ('\n=== Section 4 ===\n')
    try:
        print ('Trying to blend two images... (we are not checking the quality!)')
        sys.stdout.flush()
        im_blend = sol3.pyramid_blending(im_orig, im_orig, np.zeros((400, 400), dtype=bool), 3, 3, 5)
        print ('\tPassed!')
        print ('Checking size of blended image...')
        if im_blend.shape != im_orig.shape:
            raise ValueError('Size of blended image is different from the original images and mask used.')
        print ('\tPassed!')
    except Exception:
        print(traceback.format_exc())
        return False
    try:
        print ('Tring to call blending_example1()...')
        sys.stdout.flush()
        im1, im2, mask, im_blend = sol3.blending_example1()
        print ('\tPassed!')
        print ('Checking types of returned results...')
        if im1.dtype != np.float64:
            raise ValueError('im1 is not float64. It is %s instead.' % str(im1.dtype))
        if im2.dtype != np.float64:
            raise ValueError('im2 is not float64. It is %s instead.' % str(im2.dtype))
        if mask.dtype != bool:
            raise ValueError('mask is not bool. It is %s instead.' % str(mask.dtype))
        if im_blend.dtype != np.float64:
            raise ValueError('im_blend is not float64. It is %s instead.' % str(im_blend.dtype))
        print ('\tPassed!')
    except Exception:
        print(traceback.format_exc())
        return False

    try:
        print ('Tring to call blending_example2()...')
        sys.stdout.flush()
        im1, im2, mask, im_blend = sol3.blending_example2()
        print ('\tPassed!')
        print ('Checking types of returned results...')
        if im1.dtype != np.float64:
            raise ValueError('im1 is not float64. It is %s instead.' % str(im1.dtype))
        if im2.dtype != np.float64:
            raise ValueError('im2 is not float64. It is %s instead.' % str(im2.dtype))
        if mask.dtype != bool:
            raise ValueError('mask is not bool. It is %s instead.' % str(mask.dtype))
        if im_blend.dtype != np.float64:
            raise ValueError('im_blend is not float64. It is %s instead.' % str(im_blend.dtype))
        print ('\tPassed!')
    except Exception:
        print(traceback.format_exc())
        return False
    return True

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'dir',
    #     default=None,
    #     nargs='?',
    #     help='Dummy argument for working with the CS testing system. Has no effect.'
    # )
    # parser.add_argument(
    #     '--dev',
    #     action='store_true',
    #     help='Development mode: Assumes all student files are '
    #          'already under the directory "./current/"'
    # )
    # args = parser.parse_args()
    # if not args.dev:
    #     try:
    #         shutil.rmtree('current')
    #         shutil.rmtree('current_tmp')
    #     except Exception:
    #         pass
    #     # os.makedirs('current_tmp')
    #     # subprocess.check_call(['tar', 'xvf', sys.argv[1], '-C', 'current_tmp/'])
    #     # os.rename('current_tmp/ex3','current')
    #     shutil.rmtree('current_tmp')
    # if not os.path.isfile('current/__init__.py'):
    #     with open('current/__init__.py', 'w') as f:
    #         f.write(' ')
    # ### Supress matplotlib figures if display not available ###
    # if os.getenv('DISPLAY') is None or os.getenv('DISPLAY') == '':
    #     import matplotlib
    #     matplotlib.use('PS')
    ###########
    if presubmit():
        print('\n\n=== Presubmission Completed Successfully ===')
    else:
        print('\n\n\n !!!!!!! === Presubmission Failed === !!!!!!! ')
    print ("""\n
    Please go over the output and verify that there were no failures / warnings.
    Remember that this script tested only some basic technical aspects of your implementation.
    It is your responsibility to make sure your results are actually correct and not only
    technically valid.""")
