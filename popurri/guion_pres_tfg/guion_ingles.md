# Slide 1

Hi to everybody.
My name is Albert Ribes Marzá and today I'm presenting my project: Using Random Fourier Features with Random Forests

# Slide 2

In this project we wanted to study two problematic issues that can come up in Machine Learning. The first one is the dilemma of whether to use a linear or a nonlinear model to solve a classification or regression problem; and the other one is the difficulty of using Ensemble Learning with stable models.

Here's a brief explanation of each of them. A Machine Learning algorithm tries to generate a model from reality. And this model can be simpler or more complex. In this range of complexity there are two types of models: the linear and the nonlinear models.

Working with linear models is very desirable from an algorithmic point of view, but this models are usually too weak to explain real life data, they underfit very heavily in many applications. Nonlinear models, on contrary, are more powerful and have more flexibility, but they require more time to train a model. So many times in prior literature it has been addressed the problem of dealing with nonlinear problems with linear models.

The problem with Ensemble Learning is that, in order to successfully train an ensemble of models of the same type, you need all of them to have very little correlation between them. While this can easily be achieved with unstable models such as the Decision Tree, it is very difficult to do that with stable ones such as Support Vector Machines and Logistic Regression. This drawback prevents us from using Ensemble Learning with these models.

So, in this project we studied some ways to overcome these two problems.

# Slide 3

To do so we'll be using the feature space of a kernel function. But first we need to know what is a kernel function.

"A kernel function is a function that is equivalent to the dot product of its inputs transformed with some function phi". These functions are primarily used with Support Vector Machines, since they don't need to work directly with the data points, but just with the inner products of the data points.

SVM solve the problem of finding and hyper-plane that splits the data in two sets, one for each class. But sometimes this hyper-plane doesn't exists, due to limitations in the feature space of the data. To overcome this problems, they use a kernel function, which implicitly performs a transformation of the data to a different feature space, where an hyper-plane can be found.

But using the kernel trick has two main problems:
- First of all, it can only be used with models that only work with the inner products of the inputs, not the inputs directly.
- Secondly, when an SVM uses the kernel trick, the optimizations problem to solve becomes two costly. It is cubic with the number of instances. This cost is unacceptable for large scale datasets.

So, to try to solve the previously mentioned problems we will use an approximation of the implicit function of a particular kernel, the RBF, which stands for Radial Basis Function. This is the most used kernel function in the Machine Learning community. It has the special feature that it implicit function, phi, maps the data to a feature space with infinite dimensions, so it is not possible to compute it directly.

To generate this approximation we have used two different methods: the Random Fourier Features and the Nyström method.

# Slide 4

We have proposed two hypothesis:
- Through the usage of Random Features we can train linear models that learn from nonlinear data
- The Random Features can allow us to successfully train an ensemble of Support Vector Machine and Logistic Regression.

We claimed that because with them we can generate many different variations of the same data, and that could allow us to generate many different estimators.

To check the hypothesis we have suggested some experiments with Logistic Regression, Decision Tree and Support Vector Machine. We need to take into account that using these Random Mappings will increase the time spent in the learning process. But as the cost is linear with the number of instances, the process will be scalable to very big datasets.

# Slide 5

  There are many different ways to combine the ensemble methods with the random mappings. This project is based on Random Forest, and we've defined four different ways to mix these methods. At the beginning we expected one of them to be clearly better than the other ones, but finally all of them showed very similar results.

# Slide 6

So, lets jump to the experimental setup. We've performed several experiments in order to test the hypothesis. To do so we've used 9 different datasets, all from classification problems.

The first one was with Support Vector Machine. This model can use the kernel trick, but it is very expensive to do it. The cost is cubic with the number of instances, something unacceptable for large scale datasets.

With the first we wanted to see if we could increase the accuracy of a linear SVM with an acceptable cost, using the Random Features. Hence, the objective was not to beat an SVM using the kernel trick, but a linear SVM.

Among all the datasets that we have used, only two of them had a very big number of instances, 70000. These are the ones we will need to look at to compare the execution time

# Slide 7

Here I show the results obtained with two of the datasets. We can compare the error rate of each of the models, as well as the time required to build them. The two models in the right show the results of using each of the Random Mappings the the SVM. We are comparing the error rate with the second one, which is a single Support Vector Machine. The first one is a Support Vector Machine using the kernel trick.

When we say that the error rate is 0.20, we mean that this model predicted the wrong class for the 20% of the instances. The training error rate is the one obtained with the instances used during the training process, and the test is the one obtained with new, unseen data. This is the one we are interested in.

We can see that the models using the Random Features outperformed the linear SVM, but not the one using the kernel trick.

Here we can see the time that was spent to train each of the models. In this case it is very difficult to see it, because using the kernel trick it was several orders of magnitude higher.

# Slide 8

Here we can see what was the error decrease for each of the problems. MNIST and Fashion MNIST are the ones with 70 000 instances. We could outperform the linear SVM in all of the problems. With the "Vowel" dataset we could decrease the error rate by almost 40 points. This is a very surprising result.




<!-- This experiment has showed an increase in the accuracy for all of the datasets. For most of them we could reduce the error rate by 5%. -->


<!-- These results support the first hypothesis for Support Vector Machines. -->

# Slide 9

Here I've shown the results for other datasets. We can see that with them the difference in the training time is not so relevant. This is because they have few instances

<!-- With the rest of the datasets the difference in time is not so big, but this is because they have very few instances. -->

# Slide 10

Then we ran the same experiment with Logistic Regression. This is also a linear model, but unlike Support Vector Machines, it cannot use the kernel trick, so this is the only method we know to learn nonlinear features.

# Slide 11

With this dataset we also observed an increase of the accuracy for most of the datasets, about 3 or 4%. Again, the dataset "Vowel" shows very good results.

# Slide 12

We can see in these charts that using the random features requires more training time, but cost is still acceptable. The performance of using Random Fourier Features or Nyström is almost the same.

# Slide 13

Finally, we ran the experiment with Decision Tree. Now, this is already a non-linear model, so we didn't expect it to benefit from the usage of the Random Features.

# Slide 14

When we ran the experiment, we saw that it was not only that it hadn't improved, but it got worse. We increased the error rate for most of the problems.
<!-- The accuracy obtained decreased by 10 %. -->

The results of these three experiments support the hypothesis that linear models can benefit from using the Random Features.

# Slide 15

To check the second hypothesis, we used the Random Features with ensembles of SVMs and Logistic Regression. We wanted to see if those ensembles were able to outperform a single model using the Random Features.

# Slide 16

In these charts the firt two bars show the results using the Random Fourier Features, and the other two with Nyström. It is important to pay attention to the numbers. The error rate was decreased by less that 1% for each of the datasets. We could not see any significant improve for any of the problems.

# Slide 17

And the results where the same with Logistic Regression. We could not benefit from using an ensemble with any of the datasets. Besides, the training time with the ensembles is a lot higher, since we had to train 50 estimators instead of one.


Based on these results, There is no evidence to support our second hypothesis. It seems that the Random Features aren't enough to allow ensemble learning with these models.

# Slide 18

The conclusions of this project are that we found enough evidences to support our first hypothesis, but not for the second one, which seems to be wrong.

In addition to that, we haven't observed very big differences between using the Random Fourier Features and the Nyström method.

# Slide 19

As a future work, I propose some extensions to this project. These are ideas that appeared during the development but weren't studied due to lack of means.

First, we've been focused on solving classification problems, but there is nothing that prevents these ideas to be used with regression. It would be interesting to check the hypothesis for those.

We developed the project approximating the feature space of the RBF, but there are others kernels out there which could have been used. Maybe we could observe different results with other kernels.

And finally, we have defined the ensembles based on the Random Forest algorithm, so we only tested on ensemble method, Bagging. Maybe other methods, such as boosting, could benefit of using the Random Features.

# Slide 20

And actually I'm done. I'll be happy to answer any questions you have.
