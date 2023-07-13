import numpy as np
import statistics as stats
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

hippocsv = pd.read_csv("OASIS-hippocampus.csv")

hippocsv2 = hippocsv

#making all the 1's dementia and 0's healthy
hippocsv['Dementia'] = hippocsv['Dementia'].map({0: "Healthy", 1:"Dementia"})

hippoRight = hippocsv2["RightHippoVol"]
hippoLeft = hippocsv2["LeftHippoVol"]


# makking scatterplot
hipposcatterplot = sns.scatterplot(hippocsv2, x= hippoRight, y=hippoLeft, hue="Dementia")
plt.title("Right Hippocampus Volume vs Left Hippocampus Volume")

print(hipposcatterplot)
hipposcatterplotRight = sns.displot(data= hippocsv2, x = hippoRight, kind = "kde", hue="Dementia", rug=True)

hipposcatterplotLeft = sns.displot(data= hippocsv2, x = hippoLeft, kind = "kde", hue = "Dementia", rug=True)


newhippocsv = pd.read_csv("OASIS-hippocampus.csv")

phealthy = 0
pnothealthy = 0
for i in newhippocsv["Dementia"]:
    if i == 0:
        phealthy += 1
    else:
        pnothealthy += 1

priorhealthy = phealthy / (phealthy + pnothealthy)
priornothealthy = pnothealthy / (phealthy + pnothealthy)

# getting dementia training set (when traindata = 0) and when dementia is 1

hippo_dimentia = newhippocsv[newhippocsv["TrainData"] == 0]
hippo_dimentia = hippo_dimentia[hippo_dimentia["Dementia"] == 1]
# print(len(hippo_dimentia))


# getting the healthy training set (when traindata = 0) and when dementia 0
hippo_healthy = newhippocsv[newhippocsv["TrainData"] == 0]
hippo_healthy = hippo_healthy[hippo_healthy["Dementia"] == 0]
# print(len(hippo_healthy))

# length of whole row of testing data
testingdata = newhippocsv[newhippocsv["TrainData"] == 1]
testingdataindex = testingdata.index

# print(testingdata)
# print(testingdatalen)

# finding the mean and stdev of the the dementia class

hippo_dimentia_right = hippo_dimentia['RightHippoVol']
hippo_dimentia_left = hippo_dimentia['LeftHippoVol']
# mean of dementia
hippo_dimentia_mean_right = stats.mean(hippo_dimentia_right)
hippo_dimentia_mean_left = stats.mean(hippo_dimentia_left)
# std of dementia
hippo_dimentia_std_right = stats.stdev(hippo_dimentia_right)
hippo_dimentia_std_left = stats.stdev(hippo_dimentia_left)

hippo_healthy_right = hippo_healthy["RightHippoVol"]
hippo_healthy_left = hippo_healthy["LeftHippoVol"]
# mean of healthy
hippo_healthy_mean_right = stats.mean(hippo_healthy_right)
hippo_healthy_mean_left = stats.mean(hippo_healthy_left)
# std of dementia
hippo_healthy_std_right = stats.stdev(hippo_healthy_right)
hippo_healthy_std_left = stats.stdev(hippo_healthy_left)

correct_predictions = 0

for ind in range(len(testingdataindex)):
    LeftHippoVol = testingdata["LeftHippoVol"]
    LeftVolVal = LeftHippoVol.values[ind]
    RightHippoVol = testingdata["RightHippoVol"]
    RightVolVal = RightHippoVol.values[ind]
    DimentiaVal = testingdata["Dementia"]
    # testingdata

    # getting healthy gaussian values
    norm_healthy_right = norm(hippo_healthy_mean_right, hippo_healthy_std_right)
    norm_healthy_left = norm(hippo_dimentia_mean_left, hippo_healthy_std_left)

    # getting dimentia values
    norm_dimentia_right = norm(hippo_dimentia_mean_right, hippo_dimentia_std_right)
    norm_dimentia_left = norm(hippo_dimentia_mean_left, hippo_dimentia_std_left)

    # getting prob density function for healthy and dementia
    gauss_healthy_left = norm_healthy_left.pdf(LeftVolVal)
    gauss_healthy_right = norm_healthy_right.pdf(RightVolVal)

    gauss_dimentia_left = norm_dimentia_left.pdf(LeftVolVal)
    gauss_dimentia_right = norm_dimentia_right.pdf(RightVolVal)

    # multiplying probabilites together to find total probs
    total_healthy_gauss = gauss_healthy_left * gauss_healthy_right * priorhealthy
    total_dimentia_gauss = gauss_dimentia_left * gauss_dimentia_right * priornothealthy

    # checking whether the classifier is greater than 0.5, then check if it's right

    if ((total_dimentia_gauss / (total_dimentia_gauss + total_healthy_gauss)) > 0.5):

        # have to actually check if it's actually right or not
        if (newhippocsv["Dementia"].values[ind] == 1):
            correct_predictions += 1

    # Else if it's not those it has to be 0 so just add the counter
    else:
        correct_predictions += 1

prediction_accuracy = ((correct_predictions / (len(testingdataindex))) * 100)

print("The prediction accuracy of this Naive Bayes model is " + str(round(prediction_accuracy, 2)) + "%")

hippocsv = pd.read_csv("OASIS-hippocampus.csv")

training = hippocsv[hippocsv["TrainData"] == 1]

cleaned_training = training[["LeftHippoVol", "RightHippoVol", "Dementia"]]

scaled_features = (cleaned_training[["LeftHippoVol", "RightHippoVol"]] - cleaned_training[["LeftHippoVol", "RightHippoVol"]].min()) / (cleaned_training[["LeftHippoVol", "RightHippoVol"]].max() - cleaned_training[["LeftHippoVol", "RightHippoVol"]].min())

X_train = np.hstack((np.ones((scaled_features.shape[0], 1)), scaled_features))

y_train = cleaned_training["Dementia"].values



print(X_train)
print(y_train)


# NOTE: FORMULA FOR LOG LIKELIHOOD = y lny + (1-y)ln(1-sigmoid z)

# Define the negative log-likelihood function
def neg_log_likelihood(X, y, beta):
    z = np.dot(X, beta)

    Sigmoid_z = 1 / (1 + np.exp(-z))

    return -np.sum(y * np.log(Sigmoid_z) + (1 - y) * np.log(1 - Sigmoid_z))


# Define the gradient of the negative log-likelihood function
def gradient(X, y, beta):
    z = np.dot(X, beta)
    Sigmoid_z = 1 / (1 + np.exp(-z))

    return np.dot(X.T, (Sigmoid_z - y))


# putting 0s into beta
beta = np.zeros(X_train.shape[1])
# print(np.shape(X_train))
# set up the step size
delta = 0.01
grad = None
neg_log_likelihood_value = None
num_iterations = 3000

neg_log_likelihood_values = []

# Initialize previous negative log-likelihood
prev_neg_log_likelihood = float('inf')


def setup_vals(grad, beta, likelihood_value):
    grad = gradient(X_train, y_train, beta)
    beta -= delta * grad

    # Computing the negative log-likelihood
    likelihood_value = neg_log_likelihood(X_train, y_train, beta)

    return grad, beta, likelihood_value


def update_vals(prev_val, curr_val):
    prev_val = curr_val


for i in range(num_iterations):

    grad, beta, neg_log_likelihood_value = setup_vals(grad, beta, neg_log_likelihood_value)

    neg_log_likelihood_values.append(neg_log_likelihood_value)

    if abs(prev_neg_log_likelihood - neg_log_likelihood_value) < 1e-6:
        break

    update_vals(prev_neg_log_likelihood, neg_log_likelihood_value)

    # print(f"Iteration {i + 1}: Negative Log-Likelihood = {neg_log_likelihood_value}")

    # Print the final  values
print("Final Neg log likelihood value", neg_log_likelihood_value)

print("Beta:", beta)

print("Gradient:", grad)

plt.plot(neg_log_likelihood_values)
plt.xlabel('Iteration Number')
plt.ylabel('Negative Log-Likelihood')
plt.title('Negative Log-Likelihood vs. Iteration Number')
plt.show()

#NOTE: Formula for sigmoid and beta is as follows:  -(BETA0 + beta1 * x) / beta2 = right hippo volume

healthy_training_data = scaled_features[cleaned_training["Dementia"] == 0]
dementia_training_data = scaled_features[cleaned_training["Dementia"] == 1]

plt.scatter(healthy_training_data["LeftHippoVol"], healthy_training_data["RightHippoVol"], color="blue", label = 'Healthy')
plt.scatter(dementia_training_data["LeftHippoVol"], dementia_training_data["RightHippoVol"], color="orange", label = 'Healthy')

LeftHippoVol = dementia_training_data["LeftHippoVol"].values
# print(LeftHippoVol)
RightHippoVol = -(beta[0] + beta[1] * LeftHippoVol) / beta[2]
# print(RightHippoVol)

plt.plot(LeftHippoVol, RightHippoVol, color="purple", label="Separating Line between Healthy and Dementia")
plt.xlabel("Left Hippocampus Volume (Scaled/Adjusted)")
plt.ylabel("Right Hippocampus Volume (Scaled/Adjusted)")
plt.legend
plt.show()
plt.close()



#Taken from part a
testing_data = hippocsv[hippocsv["TrainData"] == 0]
cleaned_testing_data = testing_data[["LeftHippoVol", "RightHippoVol", "Dementia"]]
scaled_testing_features = (cleaned_testing_data[["LeftHippoVol", "RightHippoVol"]] - cleaned_training[["LeftHippoVol", "RightHippoVol"]].min()) / (cleaned_training[["LeftHippoVol", "RightHippoVol"]].max() - cleaned_training[["LeftHippoVol", "RightHippoVol"]].min())
X_testing_data = np.hstack((np.ones((scaled_testing_features.shape[0], 1)), scaled_testing_features))
y_testing_data = cleaned_testing_data["Dementia"].values



#NOTE: FORMULA FOR SIGMOID = 1/(1+e^-t)

# Make predictions using the logistic regression model
z_testing = np.dot(X_testing_data, beta)
sigmoid_z_test = 1 / (1 + np.exp(-z_testing))


y_pred = []

for value in sigmoid_z_test:
    if value > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)

correct_predictions = 0
total_predictions = len(y_testing_data)

for i in range(total_predictions):
    if y_testing_data[i] == y_pred[i]:
        correct_predictions += 1

log_accuracy = round(((correct_predictions / total_predictions) * 100), 2)

naive_bayes_acc = 74.8


final_result = """
                The final result for the logistic regression is {accuracy1} percent while the naive bayes prediction result was {accuracy2}.
                The naive bayes result was higher than my result for the logistic regression. 
                """.format(accuracy1 = log_accuracy, accuracy2 = naive_bayes_acc)
print(final_result)

