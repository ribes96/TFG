#!/usr/bin/python3

from demo_utils.hyper_params.box_black_bag.covertype import covertype_dic
from demo_utils.hyper_params.box_black_bag.digits import digits_dic
from demo_utils.hyper_params.box_black_bag.fall_detection import fall_detection_dic
from demo_utils.hyper_params.box_black_bag.mnist import mnist_dic
from demo_utils.hyper_params.box_black_bag.pen_digits import pen_digits_dic
from demo_utils.hyper_params.box_black_bag.satellite import satellite_dic
from demo_utils.hyper_params.box_black_bag.segment import segment_dic
from demo_utils.hyper_params.box_black_bag.vowel import vowel_dic


def my_join():
    ret_dic = {}
    ret_dic['covertype'] = covertype_dic()
    ret_dic['digits'] = digits_dic()
    ret_dic['fall_detection'] = fall_detection_dic()
    ret_dic['mnist'] = mnist_dic()
    ret_dic['pen_digits'] = pen_digits_dic()
    ret_dic['satellite'] = satellite_dic()
    ret_dic['segment'] = segment_dic()
    ret_dic['vowel'] = vowel_dic()

    return ret_dic
