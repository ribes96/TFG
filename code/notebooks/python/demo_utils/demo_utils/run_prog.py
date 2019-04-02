
from demo_utils.testing_code import exp1_1
from demo_utils.testing_code import generate_graphs
# from demo_utils.general import SUPPORTED_DATASETS


import matplotlib.pyplot as plt
import numpy as np


# def generate_png(exp_code, dts_name, labels, train_scores,
#                  test_scores):
#     filename = f'{exp_code}/{dts_name}'
#     o_filename = f'experimental_graphs/{filename}.png'
#     fig, (test, train) = plt.subplots(1, 2, sharey=True)
#     fig.suptitle(dts_name)
#
#     test.bar(labels, test_scores)
#     train.bar(labels, train_scores)
#
#     test.set_ylabel('Accuracy')
#     test.set_title('Test Score')
#     train.set_title('Train Score')
#
#     test.tick_params(rotation=45)
#     train.tick_params(rotation=45)
#
#     plt.savefig(o_filename)


# def fun(dts_name='digits'):
#     x = np.array([1, 2, 3, 4])
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
#
#     fig.suptitle(dts_name)

    # ax1.plot(x, x, label='linear')
    # ax2.plot(x, x, label='linear')
    # ax1.plot(x, x**2, label='quadratic')
    # ax2.plot(x, x**3, label='cubic')

    # ax1.bar(['uno','cos','tres','cuatro'], x**1)
    # ax1.bar(x, x**2, label='third')

    # ax2.bar(['siz','seis','siete','ocho'], x**3)
    # ax2.bar(x, x**4, label='quinto')

    # ax1.set_ylabel('Accuracy')
    #
    # ax1.set_title("Test Score")
    # ax2.set_title("Train Score")
    #
    # ax1.tick_params(rotation=45)
    # ax2.tick_params(rotation=45)

    # plt.setp(ax1, )
    # ax1.legend()
    # ax2.legend()

    # plt.savefig('imagen_de_prueba.png')

    # plt.show()

# for dts_name in SUPPORTED_DATASETS:
#     exp1_1(dts_name)

# exp1_1('digits')
