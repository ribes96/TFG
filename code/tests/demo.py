#!/usr/bin/python3

import sys
sys.path.append("..")
from read_data import get_data

import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
import matplotlib.pyplot as plt
import random
import os
import code
from sklearn.preprocessing import normalize
from scipy import stats
from functools import reduce
import math
from sklearn.tree import DecisionTreeClassifier


################################################################################
## Checkers
################################################################################

def check_input_range(x, options):
    if x == "": return previous_option
    if not x.isdigit() or not (0 <= int(x) < len(options)):
        inform("Invalid option. Choose a valid one (-____-)")
        return None
    else: return int(x)

def check_greater_zero(x):
    try:
        f = float(x)
        if f <= 0: raise ValueError
        return f
    except ValueError:
        inform("This value should be a number greater than zero")
        return None

def check_integer(x):
    try:
        b = float(x).is_integer()
        if not b: raise ValueError
        return int(float(x))
    except ValueError:
        inform("This value should be integer")
        return None


# def check_input(x):
#     if not x.isdigit() or not (0 <= int(x) < len(main_menu_options)):
#         # print("Invalid option. Choose a valid one (-____-)")
#         inform("Invalid option. Choose a valid one (-____-)")
#         return None
#     else: return int(x)

# def check_gamma(x):
#     try:
#         f = float(x)
#         if f <= 0:
#             inform("Gamma should be greater than 0")
#             print("Gamma should be greater than 0")
#             return None
#         else: return f
#     except ValueError:
#         inform("Gamma should be a number greater than 0")
#         print("Gamma should be a number greater than 0")
#         return None

# def check_D(x):
#     if x.isdigit() and int(x) > 0:
#         return int(x)
#     else:
#         inform("D should be an integer greater than 0")
#         print("D should be an integer greater than 0")
#         return None

# def check_param_input(x):
#     if not x.isdigit() or not (0 <= int(x) < len(params)):
#         print("Invalid option. Choose a valid one (-____-)")
#         return None
#     else: return int(x)

################################################################################
## Program showers
################################################################################
def show_noticeboard():
    for i in noticeboard:
        print(i[0])

def show_current_tuning_parameters():
    print("#####################")
    print("#  Current Tuning parameters")
    print("#  ------------------")
    print("#  gamma:", gamma)
    print("#  d:", d)
    print("#  D:", D)
    print("#  perform_normalization:",perform_normalization)
    print("#  use_offset:", use_offset)
    print("#  use_same_instance:", use_same_instance)
    print("#")
    print("#####################")

def show_main_menu():
    global previous_option
    os.system('clear')
    show_noticeboard()
    show_current_tuning_parameters()
    general_menu(main_menu_options)
    # print_options()
    a = check_input_range(input("     -------------------\n     Enter an option (default to " + str(previous_option) + ") : "), main_menu_options)
    # a = check_input(input("     -------------------\n     Enter an option: "))
    print()
    if a is not None:
        main_menu_options[a][1]()
        previous_option = a
    print("------------------------------------------")

def show_tuning_menu():
    global previous_option
    os.system('clear')
    show_noticeboard()
    show_current_tuning_parameters()
    print("What parameter do you want to tune?")
    general_menu(tuning_menu_options)
    # print_params()
    a = check_input_range(input("     -------------------\n     Enter an option (default to " + str(previous_option) + "): "), tuning_menu_options)
    # a = check_param_input(input("     -------------------\n     Enter an option: "))
    # if a == 0: break
    # else:
    if a is not None:
        tuning_menu_options[a][1]()
        previous_option = a
    print("We will change the parameter", a)

def show_gamma_menu():
    global gamma
    os.system('clear')
    show_noticeboard()
    show_current_tuning_parameters()
    a = check_greater_zero(input("Enter the new value for gamma (gamma > 0): "))
    if a is not None:
        gamma = a
        inform("gamma changed to " + str(gamma))
        change_menu_to_tune()

def show_D_menu():
    global D
    os.system('clear')
    show_noticeboard()
    show_current_tuning_parameters()
    a = check_greater_zero(input("Enter the new integer value for D (D > 0): "))
    if a is not None:
        a = check_integer(str(a))
        if a is not None:
            D = a
            inform("D changed to " + str(D))
            change_menu_to_tune()





################################################################################
## Histogram showers
################################################################################

# def see_random_weight():
#     #TODO este vector coje datos distintos
#     global current_instance
#     if not use_same_instance:
#         current_instance = random.randrange(random_weights.shape[0])
#     inform("Showing histogram of a random weight vector...")
#     plt.gcf().clear()
#     plt.hist(random_weights[current_instance])
#     plt.title("Histogram of a weight vector")
#     inform2("Showing from instance " + str(current_instance))

def see_random_input_instance():
    global current_instance
    if not use_same_instance:
        current_instance = random.randrange(train_data.shape[0])
    inform("Showing histogram of a random input instance...")
    plt.gcf().clear()
    plt.hist(train_data[current_instance])
    plt.title("Histogram of a random input instance")
    inform2("Showing from instance " + str(current_instance))
    inform_stats(train_data[current_instance])

def see_random_cos():
    global current_instance
    if not use_same_instance:
        current_instance = random.randrange(cos_projection.shape[0])
    inform("Showing histogram of a random cosine projection...")
    plt.gcf().clear()
    plt.hist(cos_projection[current_instance])
    plt.title("Histogram of the cosine projection")
    inform2("Showing from instance " + str(current_instance))
    inform_stats(cos_projection[current_instance])

def see_random_sin():
    global current_instance
    if not use_same_instance:
        current_instance = random.randrange(sin_projection.shape[0])
    inform("Showing histogram of a random sine projection...")
    plt.gcf().clear()
    plt.hist(sin_projection[current_instance])
    plt.title("Histogram of the sine projection")
    inform2("Showing from instance " + str(current_instance))
    inform_stats(sin_projection[current_instance])

def see_random_sin_cos():
    global current_instance
    if not use_same_instance:
        current_instance = random.randrange(sin_cos_projection.shape[0])
    inform("Showing histogram of a random sine and cosine projection...")
    plt.gcf().clear()
    plt.hist(sin_cos_projection[current_instance])
    plt.title("Histogram of the joined sine and cosine projection")
    inform2("Showing from instance " + str(current_instance))
    inform_stats(sin_cos_projection[current_instance])



def see_final_projection():
    global current_instance
    if not use_same_instance:
        current_instance = random.randrange(final_projection.shape[0])
    inform("Showing histogram of a random instance of the final projection...")
    plt.gcf().clear()
    plt.hist(final_projection[current_instance])
    plt.title("Histogram of the final projection")
    inform2("Showing from instance " + str(current_instance))
    inform_stats(final_projection[current_instance])

# def see_dot_product():
#     global current_instance
#     if not use_same_instance:
#         current_instance = random.randrange(dot_product.shape[0])
#     inform("Showing histogram of a random instance of dot product...")
#     plt.gcf().clear()
#     plt.hist(dot_product[current_instance])
#     plt.title("Histogram of the dot product")
#     inform2("Showing from instance " + str(current_instance))
#     inform_stats(dot_product[current_instance])

################################################################################
## Menu printers
################################################################################

################################################################################
## Tuners
################################################################################
def toggle_perform_normalization():
    global perform_normalization
    perform_normalization = not perform_normalization
    inform("perform_normalization changed to " + str(perform_normalization))

def toggle_use_offset():
    global use_offset
    use_offset = not use_offset
    inform("use_offset changed to " + str(use_offset))

def toggle_use_same_instance():
    global use_same_instance
    use_same_instance = not use_same_instance
    inform("use_same_instance changed to " + str(use_same_instance))
# def tune_gamma():
#     show_noticeboard()
#     global gamma
#     # print("Enter the new value for gamma (gamma > 0)")
#     a = None
#     while a is None:
#         a = check_greater_zero(input("Enter the new value for gamma (gamma > 0): "))
#     gamma = a

# def tune_D():
#     show_noticeboard()
#     global D
#     a = None
#     while a is None:
#         a = check_D(input("Enter the new value for D (D > 0): "))
#     D = a

################################################################################
## Menu changers
################################################################################
def change_menu_to_tune():
    global menu_index
    # This is hardcoded
    menu_index = 1

def change_menu_to_main():
    global menu_index
    menu_index = 0
    update_data()

def change_menu_to_gamma_tune():
    global menu_index
    menu_index = 2

def change_menu_to_D_tune():
    global menu_index
    menu_index = 3

################################################################################
## Utils
################################################################################
def update_data():
    global train_data
    global random_weights
    global random_offset
    global dot_product
    global dot_product_mod2pi
    global cos_projection
    global sin_projection
    global sin_cos_projection
    global sin_final_projection
    global cos_final_projection
    global final_projection

    if perform_normalization:
        train_data = normalize(train_data)
    if use_offset:
        random_offset = np.random.uniform(0, 2 * np.pi, D)
    else:
        random_offset = np.random.uniform(0, 0, D)
    random_weights = (np.sqrt(2 * gamma) * np.random.normal(size=(d, D)))
    dot_product = safe_sparse_dot(train_data, random_weights)
    dot_product += random_offset
    dot_product_mod2pi = dot_product % (2 * np.pi)
    cos_projection = np.cos(dot_product)
    sin_projection = np.sin(dot_product)
    sin_cos_projection = np.concatenate((sin_projection, cos_projection), axis = 1)
    sin_final_projection = sin_projection * np.sqrt(2.) / np.sqrt(D)
    cos_final_projection = cos_projection * np.sqrt(2.) / np.sqrt(D)
    final_projection = sin_cos_projection * np.sqrt(2.) / np.sqrt(2*D)
    inform("Data has been updated!")



def inform(st):
    global info
    noticeboard[1][0] = st

def inform2(st):
    global info
    noticeboard[2][0] = st

def inform3(st):
    global info
    noticeboard[3][0] = st

# Display the menu specified
def general_menu(menu):
    for i in range(len(menu)):
        print(i,": ",menu[i][0])

################################################################################
## Constant arrays
################################################################################

def change_menu_to_interactive():
    # inform("This option is currently not available")
    # change_menu_to_main()
    # patata = "túmás"
    # code.interact(local=locals())
    # global gamma
    code.interact(local=dict(globals(), **locals()))


# def see_random_weight():
#     #TODO este vector coje datos distintos
#     global current_instance
#     if not use_same_instance:
#         current_instance = random.randrange(random_weights.shape[0])
#     inform("Showing histogram of a random weight vector...")
#     plt.gcf().clear()
#     plt.hist(random_weights[current_instance])
#     plt.title("Histogram of a weight vector")
#     inform2("Showing from instance " + str(current_instance))

def see_row_dot_product_mod2pi():
    # TODO lo de use_same_instance
    c = random.randrange(dot_product_mod2pi.shape[0])
    inform("Showing a histogram of a random row of dow_product_mod2pi...")
    plt.gcf().clear()
    plt.hist(dot_product_mod2pi[c], bins = 30)
    plt.title("Histogram of a random row of dot_product_mod2pi")
    inform2("Showing the row " + str(c))
    inform_stats(dot_product_mod2pi[c])

def see_column_dot_product_mod2pi():
    # TODO lo de use_same_instance
    c = random.randrange(dot_product_mod2pi.shape[1])
    inform("Showing a histogram of a random column of dot_product_mod2pi...")
    plt.gcf().clear()
    plt.hist(dot_product_mod2pi[:,c], bins = 30)
    plt.title("Histogram of a random column of dot_product_mod2pi")
    inform2("Showing the column " + str(c))
    inform_stats(dot_product_mod2pi[:,c])

def see_row_dot_product():
    # TODO lo de use_same_instance
    c = random.randrange(dot_product.shape[0])
    inform("Showing a histogram of a random row of dot_product...")
    plt.gcf().clear()
    plt.hist(dot_product[c])
    plt.title("Histogram of a random row of dot_product")
    inform2("Showing the row " + str(c))
    inform_stats(dot_product[c])

def see_column_dot_product():
    # TODO lo de use_same_instance
    c = random.randrange(dot_product.shape[1])
    inform("Showing a histogram of a random column of dow_product...")
    plt.gcf().clear()
    plt.hist(dot_product[:,c])
    plt.title("Histogram of a random column of dot_product")
    inform2("Showing the column " + str(c))
    inform_stats(dot_product[:,c])


def see_W_row():
    # TODO lo de use_same_instance
    c = random.randrange(random_weights.shape[0])
    inform("Showing a histogram of a random row of W...")
    plt.gcf().clear()
    plt.hist(random_weights[c])
    plt.title("Histogram of a random row of W")
    inform2("Showing the row " + str(c))
    inform_stats(random_weights[c])
    # st = get_stats(random_weights[c])
    # inform3(reduce((lambda x,y: x + "\n" + y), list_2_columns(st)))

def inform_stats(dat):
    # get_stats return a list, each element one string to show
    st = get_stats(dat)
    inform3(reduce((lambda x,y: x + "\n" + y), list_2_columns(st)))


def see_W_column():
    # TODO lo de use_same_instance
    c = random.randrange(random_weights.shape[1])
    inform("Showing a histogram of a random column of W...")
    plt.gcf().clear()
    plt.hist(random_weights[:,c])
    plt.title("Histogram of a random column of W")
    inform2("Showing the column " + str(c))
    inform_stats(random_weights[:,c])

def get_stats(dat):
    st = stats.describe(dat)
    # dat2 = stats.zscore(dat)
    # st2 = stats.describe(dat2)
    ret = []
    ret.append("nobs: " + str(st.nobs))
    ret.append("min: " + str(st.minmax[0]))
    ret.append("max: " + str(st.minmax[1]))
    ret.append("mean: " + str(st.mean))
    ret.append("variance: " + str(st.variance))
    ret.append("std. dev: " + str(np.sqrt(st.variance)))
    # ret.append("std. dev: " + str(np.sqrt(st.variance)))
    # ret.append("z-score variance: " + str(st2.variance))
    # ret.append("z-score mean: " + str(st2.mean))
    return ret

def list_2_columns(l):
    # half = int(len(l) / 2)
    half = math.ceil(len(l) / 2)
    l1 = l[:half]
    l2 = l[half:]
    maxlength = len(max(l1, key = len))
    ret = []
    for i, j in zip(l1, l2):
        ret.append(i.ljust(maxlength) + "   " + j)
    return ret

def perform_map(dat):


def see_score_cos():
    # TODO
    tree = DecisionTreeClassifier()
    tree.fit(cos_final_projection, train_pred)
    sc = tree.score(cos_final_projection, train_pred)
    inform2("The score was " + str(sc))

def see_score_sin_cos():
    # TODO
    h = 2

main_menu_options = [
["Exit the program\n     -----------------", exit],
["Tune parameters", change_menu_to_tune],

# ["See histogram of a random weight vector [X]", see_random_weight],

["See histogram of a random row of matrix W", see_W_row],
["See histogram of a random column of matrix W", see_W_column],

["See histogram of a random input instance", see_random_input_instance],
# ["See histogram of a random from dot_product", see_dot_product],

["See histogram of a random row of matrix dot_product", see_row_dot_product],
["See histogram of a random column of matrix dot_product", see_column_dot_product],

["See histogram of a random row of matrix dot_product_mod2pi", see_row_dot_product_mod2pi],
["See histogram of a random column of matrix dot_product_mod2pi", see_column_dot_product_mod2pi],

["See histogram of a random sin", see_random_sin],
["See histogram of a random cos", see_random_cos],

["See histogram of a random sin_cos", see_random_sin_cos],
["See histogram of a random final projection", see_final_projection],
["Enter interactive mode", change_menu_to_interactive],
# ["See score using current parameters (just cos)", see_score_cos],
# ["See score using current parameters (with sin and cos)", see_score_sin_cos],
]

tuning_menu_options = [
["Finish tuning parameters\n     ------------------------", change_menu_to_main],
["Change gamma", change_menu_to_gamma_tune],
["Change D", change_menu_to_D_tune],
["Toggle perform_normalization", toggle_perform_normalization],
["Toggle use_offset", toggle_use_offset],
["Toggle use_same_instance", toggle_use_same_instance]
]

noticeboard = [
["General information"],
[""],
[""],
[""],
["___________________"],
]

menus = [
show_main_menu,
show_tuning_menu,
show_gamma_menu,
show_D_menu,
]



## Execution starts here!

# Get the data
train_data, train_pred, test_data, test_pred = get_data()
train_pred = train_pred.ravel()
test_pred = test_pred.ravel()


# Set the default values
menu_index = 0
gamma = 1
d = train_data.shape[1]
D = 1000
perform_normalization = False
use_offset = False
use_same_instance = False
current_instance = 0
previous_option = 0

# Just to have them declared
random_weights = None
random_offset = None
dot_product = None
dot_product_mod2pi = None
cos_projection = None
sin_projection = None
sin_cos_projection = None
sin_final_projection = None
cos_final_projection = None
final_projection = None

update_data()

plt.ion()
while True:
    menus[menu_index]()
