"""
Created on Tue Mar 10 19:10:34 2020

@author: Ivan Garcia Herrera
"""
import seaborn as sns
from operator import itemgetter
import csv
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy


def loadData(filename):
    fp = open(filename, 'r')
    reader = csv.reader(fp)
    data = []
    index = 0
    for row in reader:
        if index > 0:
            data.append([int(row[0]), int(row[1]), int(row[2]), int(
                row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7])])
        index += 1

    fp.close()
    return data


def plotdata(data, labels=None, name="Default"):
    fig, ax = plt.subplots()
    if labels is None:
        plt.scatter([row[0] for row in data], [row[1] for row in data])
    else:
        plt.scatter([row[0] for row in data], [row[1]
                                               for row in data], c=labels)
    ax.grid(True)
    fig.tight_layout()
    plt.title(name)
    plt.show()


def getTotalPricePerCustomer(data):
    total = []
    index = 0
    for d in data:
        total.append([index, d[2]+d[3]+d[4]+d[5]+d[6]+d[7]])
        index += 1
    return total


def percentageRepresentMostOfIncome(sortd, percentage, income=0.8):
    total_money = sum(s[1] for s in sortd)
    # Get the percentage% of best customers
    best_clients = sortd[:int(len(sortd) * percentage / 100)]
    # Calculate the total expenditure of those customers
    best_expenditure = sum(bc[1] for bc in best_clients)
    return best_expenditure / total_money >= income


"""
    This function finds the Gaussian distribution parameters (mean and variance) of the data
    Since we want to see the points that are farthest from all others, we will use a single 
    Gaussian function for all data.
"""
def estimateGaussian(X):
    m = X.shape[0]
    # Get the mean of the given data
    data_sum = np.sum(X, axis=0)
    mean = data_sum / m
    # Get the variance of the given data
    variance = np.var(X, axis=0)
    return mean, variance


"""
    Calculate the probabilities for each point to belong to the group, depending on the Gaussian 
    function with a certain mean and variance.
             k                    
            ===                   
            \                     
    p(x) =  /   ϕ_i * N(x | μ_i, σ_i)
            ===                   
           i = 1  
                                       /          2 
                            1          |-(x - μ_i) |
    N(x | μ_i σ_i) = ----------- * exp |-----------|
                            _____      |        2  |
                    σ_i * \/ 2π        \ 2 * σ_i   /
"""
def multivariateGaussian(X, mean, variance):
    k = len(mean)
    variance = np.diag(variance)
    X = X - mean.T
    probability = 1 / ((2 * np.pi)**(k/2) * (np.linalg.det(variance)**0.5)) * \
        np.exp(-0.5 * np.sum(X @ np.linalg.pinv(variance) * X, axis=1))
    return probability


"""
    This function detects the dataset outliers and displays them on a graph
"""
def detectOutliers(data, labels, remove=False):
    copy_data = copy.deepcopy(data)
    fig, ax = plt.subplots()

    # Get the mean and the variance of the Gaussian function
    dataset = np.array(data)
    mu, var = estimateGaussian(dataset)
    p = multivariateGaussian(dataset, mu, var)

    # Draw all the points. Depending on their probability of belonging to the
    # cluster, their color will change.
    ax.scatter(
        dataset[:, 0], dataset[:, 1], marker="x", c=p, cmap='viridis')
    # The outliers will be those that have a near-zero chance of belonging to the cluster
    epsilon = 5e-09
    outliers = np.nonzero(p < epsilon)[0]

    # Remove outliers
    if remove:
        for outlier in sorted(outliers, reverse=True):
            del copy_data[outlier]
        
    # Round off the points that are outliers
    ax.scatter(
        dataset[outliers, 0], dataset[outliers, 1], marker="o", facecolor="none", edgecolor="r", s=70)
    plt.show()
    return copy_data


def expectation_maximization(data_cleaned, n_clusters):
    # Calculate the EM algorithm (according to the ideal number of clusters,
    # calculated by means of silhouette coefficient)
    em = GaussianMixture(
        n_components=n_clusters, covariance_type='full', init_params='kmeans')
    em.fit(data_cleaned)
    labels = em.predict(data_cleaned)
    plotdata(data_cleaned, labels, "EM")
    return labels


def silhouette(data_cleaned, plot=False):
    silhouettes = []
    silhouettes_data = []

    # Calculate the EM algorithm for k=2..20 and get the silhouette value for each iteration
    for i in range(2, 20):
        em = GaussianMixture(
            n_components=i, covariance_type='full', init_params='kmeans')
        em.fit(data_cleaned)
        labels = em.predict(data_cleaned)
        silhouettes.append(metrics.silhouette_score(data_cleaned, labels))
        silhouettes_data.append([i, metrics.silhouette_score(data_cleaned, labels)])

    # Get the best silhouette value for this product. Add 2 to the index obtained,
    # because the number of clusters start at 2
    best_clusters = silhouettes.index(max(silhouettes)) + 2

    if plot:
        # Plot Silhouette
        plotdata(silhouettes_data, name="Silhouettes")

    return best_clusters


def getRepresentants(data_cleaned, labels, number_of_clusters):
    for index in range(number_of_clusters):
        # Get the indexes of the customers, for each group
        indexes = [i for i, x in enumerate(labels) if x == index]
        # Get the customers of each group
        customers_of_group = [dataset[i] for i in indexes]
        customer_expenses = [d[1] for d in customers_of_group]
        # Get the mean of the expenses
        mean = sum(customer_expenses) / len(customer_expenses)

        # Get the customer whose expenses are closest to the mean of the expenses, for each group
        def difference_function(data_val): return abs(data_val - mean)
        closest_customer = min(customer_expenses, key=difference_function)
        print("The representant of cluster {} is customer: {}".format(
            index + 1, customer_expenses.index(closest_customer) + 1))


########################
##### Milestone 1 ######
########################

# Read the dataset
real_labels = ["Channel", "Region", "Fresh", "Milk",
               "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
data = loadData("./clusterdata.csv")
labels_true = [row[0] for row in data]

dataset = []
for row in range(len(data)):
    ind = 0
    for column in data[row]:
        if ind >= 2:
            dataset.append([row, column])
        ind += 1

# Detect outliers, remove them from the data and print them
dataset = detectOutliers(dataset, real_labels, True)

# Get the total expenditure per customer
total = getTotalPricePerCustomer(data)

# Identify best clients
# Sort customers by total expenditure (descending order)
sortd = sorted(total, key=itemgetter(1), reverse=True)
differences = []
for percentage in range(100):
    differences.append(
        [percentage, percentageRepresentMostOfIncome(sortd, percentage, 0.8)])

# Get the lowest percentage of best customers who represent 80% of income
lowest = next((d for d in differences if d[1] == True), None)
print("{}% of customers generate 80% of income".format(lowest[0]))
# Get those "best customers"
best_customers = sortd[:int(len(sortd) * lowest[0] / 100)]
best_customers_labels = []
for cust in total:
    best_customers_labels.append(
        1) if cust in best_customers else best_customers_labels.append(0)
# Print the relation between the best customers and the rest of them
plotdata(total, best_customers_labels,
         "Best Customers vs Rest of the customers")

# Get the ideal number of clusters, according to silhouette coefficient
n_clusters = silhouette(dataset, False)
# Expectation - Maximization
labels = expectation_maximization(dataset, n_clusters)

########################
##### Milestone 2 ######
########################

# Get the representant of each group
getRepresentants(dataset, labels, n_clusters)

customers_group1 = []
customers_group2 = []
for index in range(len(labels)):
    if labels[index] == 0:
        customers_group1.append(dataset[index])
    else:
        customers_group2.append(dataset[index])

# Obtain each group's contribution to the company
print("Contribution of group 1 (size={}) to the company: {}€".format(len(customers_group1), sum([x[1] for x in customers_group1])))
print("Contribution of group 2 (size={}) to the company: {}€".format(len(customers_group2), sum([x[1] for x in customers_group2])))

# Obtain the average income of each group
mean_group1 = sum([x[1] for x in customers_group1]) / len(customers_group1)
mean_group2 = sum([x[1] for x in customers_group2]) / len(customers_group2)
print("Average income of group 1: {}€".format(mean_group1))
print("Average income of group 2: {}€".format(mean_group2))

# In which category do group members spend the most?
real_labels_group1 = []
total_spend_per_product_group1 = []
real_labels_group2 = []
total_spend_per_product_group2 = []

# Obtain group members' spending categories
for value in customers_group1:
    product_index = data[value[0]].index(value[1])
    real_labels_group1.append(real_labels[product_index])
    total_spend_per_product_group1.append([real_labels[product_index], value[1]])

for value in customers_group2:
    product_index = data[value[0]].index(value[1])
    real_labels_group2.append(real_labels[product_index])
    total_spend_per_product_group2.append([real_labels[product_index], value[1]])

from collections import Counter
categories_group1 = Counter(real_labels_group1)
categories_group2 = Counter(real_labels_group2)
#print(categories_group1, categories_group2)

# Get total expenses per group
expenses_group1 = {}
expenses_group2 = {}
for val in total_spend_per_product_group1:
    if val[0] in expenses_group1.keys():
        expenses_group1[val[0]] += int(val[1])
    else:
        expenses_group1[val[0]] = int(val[1])

for val in total_spend_per_product_group2:
    if val[0] in expenses_group2.keys():
        expenses_group2[val[0]] += int(val[1])
    else:
        expenses_group2[val[0]] = int(val[1])
        
#print(expenses_group1, expenses_group2)

# Statistical study
# Kruskal-Wallis: Check whether the null hypothesis: "The average expenditure for each group is the same" is true
from scipy.stats.mstats import kruskal
from scipy.stats.mstats import mannwhitneyu
st, pvalue = kruskal([x[1] for x in customers_group1], [x[1] for x in customers_group2])  # x[1] contains the expenditure of each product
if pvalue < 0.05:
    print("The null hypothesis: \n\t'The average expenditure for each group is the same'\nis False")
    # Mann-Whitney for each pair of groups. Eg.: milk_group1 - milk_group2, delicatessen_group1 - delicatessen_group2
    for product in range(2, 8):
        product_group1 = []
        product_group2 = []
        # Get the expenditures per type of product
        for value in customers_group1:
            product_index = data[value[0]].index(value[1])
            if product_index == product:
                product_group1.append(value[1])
        for value in customers_group2:
            product_index = data[value[0]].index(value[1])
            if product_index == product:
                product_group2.append(value[1])
        
        st, pvalue = mannwhitneyu(product_group1, product_group2)
        if pvalue < 0.05:
            print("The null hypothesis: \n\t'The average income in the {} products is similar in groups 1 and 2'\nis False".format(real_labels[product]))
        else:
            print("The null hypothesis: \n\t'The average income in the {} products is similar in groups 1 and 2'\nis True".format(real_labels[product]))