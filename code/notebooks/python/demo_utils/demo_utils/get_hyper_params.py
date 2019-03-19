# from new_hyper_params.segment.none import get_segment_none_dic
from demo_utils.new_hyper_params.covertype.none import get_covertype_none_dic
from demo_utils.new_hyper_params.digits.none import get_digits_none_dic
from demo_utils.new_hyper_params.fall_detection.none import get_fall_detection_none_dic
from demo_utils.new_hyper_params.mnist.none import get_mnist_none_dic
from demo_utils.new_hyper_params.pen_digits.none import get_pen_digits_none_dic
from demo_utils.new_hyper_params.satellite.none import get_satellite_none_dic
from demo_utils.new_hyper_params.segment.none import get_segment_none_dic
from demo_utils.new_hyper_params.vowel.none import get_vowel_none_dic

from demo_utils.new_hyper_params.covertype.grey_ens import get_covertype_grey_ens_dic
from demo_utils.new_hyper_params.digits.grey_ens import get_digits_grey_ens_dic
from demo_utils.new_hyper_params.fall_detection.grey_ens import get_fall_detection_grey_ens_dic
from demo_utils.new_hyper_params.mnist.grey_ens import get_mnist_grey_ens_dic
from demo_utils.new_hyper_params.pen_digits.grey_ens import get_pen_digits_grey_ens_dic
from demo_utils.new_hyper_params.satellite.grey_ens import get_satellite_grey_ens_dic
from demo_utils.new_hyper_params.segment.grey_ens import get_segment_grey_ens_dic
from demo_utils.new_hyper_params.vowel.grey_ens import get_vowel_grey_ens_dic

from demo_utils.new_hyper_params.covertype.grey_bag import get_covertype_grey_bag_dic
from demo_utils.new_hyper_params.digits.grey_bag import get_digits_grey_bag_dic
from demo_utils.new_hyper_params.fall_detection.grey_bag import get_fall_detection_grey_bag_dic
from demo_utils.new_hyper_params.mnist.grey_bag import get_mnist_grey_bag_dic
from demo_utils.new_hyper_params.pen_digits.grey_bag import get_pen_digits_grey_bag_dic
from demo_utils.new_hyper_params.satellite.grey_bag import get_satellite_grey_bag_dic
from demo_utils.new_hyper_params.segment.grey_bag import get_segment_grey_bag_dic
from demo_utils.new_hyper_params.vowel.grey_bag import get_vowel_grey_bag_dic


from demo_utils.new_hyper_params.covertype.black_bag import get_covertype_black_bag_dic
from demo_utils.new_hyper_params.digits.black_bag import get_digits_black_bag_dic
from demo_utils.new_hyper_params.fall_detection.black_bag import get_fall_detection_black_bag_dic
from demo_utils.new_hyper_params.mnist.black_bag import get_mnist_black_bag_dic
from demo_utils.new_hyper_params.pen_digits.black_bag import get_pen_digits_black_bag_dic
from demo_utils.new_hyper_params.satellite.black_bag import get_satellite_black_bag_dic
from demo_utils.new_hyper_params.segment.black_bag import get_segment_black_bag_dic
from demo_utils.new_hyper_params.vowel.black_bag import get_vowel_black_bag_dic


from demo_utils.new_hyper_params.covertype.none_sigest import get_covertype_none_sigest_dic
from demo_utils.new_hyper_params.digits.none_sigest import get_digits_none_sigest_dic
from demo_utils.new_hyper_params.fall_detection.none_sigest import get_fall_detection_none_sigest_dic
from demo_utils.new_hyper_params.mnist.none_sigest import get_mnist_none_sigest_dic
from demo_utils.new_hyper_params.pen_digits.none_sigest import get_pen_digits_none_sigest_dic
from demo_utils.new_hyper_params.satellite.none_sigest import get_satellite_none_sigest_dic
from demo_utils.new_hyper_params.segment.none_sigest import get_segment_none_sigest_dic
from demo_utils.new_hyper_params.vowel.none_sigest import get_vowel_none_sigest_dic

dic_box_funcs = {
    ('covertype', 'none'): get_covertype_none_dic,
    ('digits', 'none'): get_digits_none_dic,
    ('fall_detection', 'none'): get_fall_detection_none_dic,
    ('mnist', 'none'): get_mnist_none_dic,
    ('pen_digits', 'none'): get_pen_digits_none_dic,
    ('satellite', 'none'): get_satellite_none_dic,
    ('segment', 'none'): get_segment_none_dic,
    ('vowel', 'none'): get_vowel_none_dic,

    ('covertype', 'grey_ens'): get_covertype_grey_ens_dic,
    ('digits', 'grey_ens'): get_digits_grey_ens_dic,
    ('fall_detection', 'grey_ens'): get_fall_detection_grey_ens_dic,
    ('mnist', 'grey_ens'): get_mnist_grey_ens_dic,
    ('pen_digits', 'grey_ens'): get_pen_digits_grey_ens_dic,
    ('satellite', 'grey_ens'): get_satellite_grey_ens_dic,
    ('segment', 'grey_ens'): get_segment_grey_ens_dic,
    ('vowel', 'grey_ens'): get_vowel_grey_ens_dic,

    ('covertype', 'grey_bag'): get_covertype_grey_bag_dic,
    ('digits', 'grey_bag'): get_digits_grey_bag_dic,
    ('fall_detection', 'grey_bag'): get_fall_detection_grey_bag_dic,
    ('mnist', 'grey_bag'): get_mnist_grey_bag_dic,
    ('pen_digits', 'grey_bag'): get_pen_digits_grey_bag_dic,
    ('satellite', 'grey_bag'): get_satellite_grey_bag_dic,
    ('segment', 'grey_bag'): get_segment_grey_bag_dic,
    ('vowel', 'grey_bag'): get_vowel_grey_bag_dic,

    ('covertype', 'black_bag'): get_covertype_black_bag_dic,
    ('digits', 'black_bag'): get_digits_black_bag_dic,
    ('fall_detection', 'black_bag'): get_fall_detection_black_bag_dic,
    ('mnist', 'black_bag'): get_mnist_black_bag_dic,
    ('pen_digits', 'black_bag'): get_pen_digits_black_bag_dic,
    ('satellite', 'black_bag'): get_satellite_black_bag_dic,
    ('segment', 'black_bag'): get_segment_black_bag_dic,
    ('vowel', 'black_bag'): get_vowel_black_bag_dic,

    ('covertype', 'none_sigest'): get_covertype_none_sigest_dic,
    ('digits', 'none_sigest'): get_digits_none_sigest_dic,
    ('fall_detection', 'none_sigest'): get_fall_detection_none_sigest_dic,
    ('mnist', 'none_sigest'): get_mnist_none_sigest_dic,
    ('pen_digits', 'none_sigest'): get_pen_digits_none_sigest_dic,
    ('satellite', 'none_sigest'): get_satellite_none_sigest_dic,
    ('segment', 'none_sigest'): get_segment_none_sigest_dic,
    ('vowel', 'none_sigest'): get_vowel_none_sigest_dic,
}


def get_func_call(dts_name, box_name):
    return dic_box_funcs[(dts_name, box_name)]
    # if (dts_name, box_name) == ('covertype', 'none'):
    #     return get_covertype_none_dic
    # if (dts_name, box_name) == ('digits', 'none'):
    #     return get_digits_none_dic
    # if (dts_name, box_name) == ('fall_detection', 'none'):
    #     return get_fall_detection_none_dic
    # if (dts_name, box_name) == ('mnist', 'none'):
    #     return get_mnist_none_dic
    # if (dts_name, box_name) == ('pen_digits', 'none'):
    #     return get_pen_digits_none_dic
    # if (dts_name, box_name) == ('satellite', 'none'):
    #     return get_satellite_none_dic
    # if (dts_name, box_name) == ('segment', 'none'):
    #     return get_segment_none_dic
    # if (dts_name, box_name) == ('vowel', 'none'):
    #     return get_vowel_none_dic
    #
    # if (dts_name, box_name) == ('covertype', 'grey_ens'):
    #     return get_covertype_grey_ens_dic
    # if (dts_name, box_name) == ('digits', 'grey_ens'):
    #     return get_digits_grey_ens_dic
    # if (dts_name, box_name) == ('fall_detection', 'grey_ens'):
    #     return get_fall_detection_grey_ens_dic
    # if (dts_name, box_name) == ('mnist', 'grey_ens'):
    #     return get_mnist_grey_ens_dic
    # if (dts_name, box_name) == ('pen_digits', 'grey_ens'):
    #     return get_pen_digits_grey_ens_dic
    # if (dts_name, box_name) == ('satellite', 'grey_ens'):
    #     return get_satellite_grey_ens_dic
    # if (dts_name, box_name) == ('segment', 'grey_ens'):
    #     return get_segment_grey_ens_dic
    # if (dts_name, box_name) == ('vowel', 'grey_ens'):
    #     return get_vowel_grey_ens_dic
    #
    # if (dts_name, box_name) == ('covertype', 'grey_bag'):
    #     return get_covertype_grey_bag_dic
    # if (dts_name, box_name) == ('digits', 'grey_bag'):
    #     return get_digits_grey_bag_dic
    # if (dts_name, box_name) == ('fall_detection', 'grey_bag'):
    #     return get_fall_detection_grey_bag_dic
    # if (dts_name, box_name) == ('mnist', 'grey_bag'):
    #     return get_mnist_grey_bag_dic
    # if (dts_name, box_name) == ('pen_digits', 'grey_bag'):
    #     return get_pen_digits_grey_bag_dic
    # if (dts_name, box_name) == ('satellite', 'grey_bag'):
    #     return get_satellite_grey_bag_dic
    # if (dts_name, box_name) == ('segment', 'grey_bag'):
    #     return get_segment_grey_bag_dic
    # if (dts_name, box_name) == ('vowel', 'grey_bag'):
    #     return get_vowel_grey_bag_dic
    #
    # if (dts_name, box_name) == ('covertype', 'black_bag'):
    #     return get_covertype_black_bag_dic
    # if (dts_name, box_name) == ('digits', 'black_bag'):
    #     return get_digits_black_bag_dic
    # if (dts_name, box_name) == ('fall_detection', 'black_bag'):
    #     return get_fall_detection_black_bag_dic
    # if (dts_name, box_name) == ('mnist', 'black_bag'):
    #     return get_mnist_black_bag_dic
    # if (dts_name, box_name) == ('pen_digits', 'black_bag'):
    #     return get_pen_digits_black_bag_dic
    # if (dts_name, box_name) == ('satellite', 'black_bag'):
    #     return get_satellite_black_bag_dic
    # if (dts_name, box_name) == ('segment', 'black_bag'):
    #     return get_segment_black_bag_dic
    # if (dts_name, box_name) == ('vowel', 'black_bag'):
    #     return get_vowel_black_bag_dic


def get_hyper_params(dts_name, box_name, model_name, sampler_name):
    '''
    Return a dictionary with the suitable params

    Parameters
    ----------
    dts_name : str
        'segment', 'covertype', 'digits', 'fall_detection', 'mnist',
        'pen_digits', 'satellite' or 'vowel'
    box_name : str
        'none', 'black_bag', 'grey_bag' or 'grey_ens'
    model_name : str
        'dt', 'logit' or 'linear_svc'
    sampler_name : str
        'identity', 'rff', 'nystroem'
    '''
    call_func = get_func_call(dts_name=dts_name, box_name=box_name)

    dic = call_func()
    model_dic = dic[model_name]
    sampler_dic = model_dic[sampler_name]

    return sampler_dic
