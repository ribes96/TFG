#!/usr/bin/python3

from custom_classes.ribes_RFFSampler import ribes_RFFSampler
import numpy as np

# print("Hola que tal")
datasize = (10, 3)
data = np.ones(datasize)
# print(data)
samp = ribes_RFFSampler(n_components = 2, random_state = 4);
samp.fit(data)
t = samp.transform(data)
# print("-----------")
print(t)
