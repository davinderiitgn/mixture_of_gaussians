import csv
import numpy
import random
from scipy.stats import multivariate_normal as mvn

def get_data(name,maxi=0,mini=0):	
	data = []
	with open(name, 'r') as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			for i in range(len(row)):
				row[i] = float(row[i])
				if row[i] > maxi:
					maxi = row[i]
				if row[i] < mini:
					mini = row[i]
			data = data + [row]
	return (data,maxi,mini)

def random_array(n,a,b):
	ret = [0]*n
	for i in range(n):
		ret[i] = random.uniform(a,b)
	return ret

def random_matrix(m,n,a,b):
	ret = [0]*m
	for i in range(m):
		ret[i] = random_array(n,a,b)
	return ret

def identity_3d(m,n):
	ret = [0]*m
	for i in range(m):
		ret[i] = numpy.identity(n)
	return ret

def my_sum_vector(a,b):
	c = [0]*len(a)
	for i in range(len(a)):
		c[i] = a[i]+b[i]
	return c

def my_sum_matrix(a,b):
	c = [0]*len(a)
	for i in range(len(c)):
		c[i] = [0]*len(a[0])
	for i in range(len(a)):
		for j in range(len(a[0])):
			c[i][j] = a[i][j] + b[i][j]
	return c

def my_subtract(a,b):
	c = [0]*len(a)
	for i in range(len(a)):
		c[i] = a[i]-b[i]
	return c


maxi = -1000000000
mini = 1000000000
k_min = 1
k_max = 5
data,maxi,mini = get_data('data_ul.csv',maxi,mini)

max_iter = 50

def train():

	k_final = 0
	phi_final = 0
	mean_final = 0
	sigma_final = 0
	sop = 0

	for k in range(k_min,k_max):
		w_s = [0]*len(data)
		for i in range(len(w_s)):
			w_s[i] = [0]*k

		phi = random_array(k,0,1)
		sum_temp = sum(phi)
		for i in range(len(phi)):
			phi[i] = phi[i]/sum_temp

		mean = random_matrix(k,len(data[0]),mini,maxi)

		sigma = identity_3d(k,len(data[0]))

		for _ in range(max_iter):

			# E-STEP
			for i in range(len(data)):
				temp = [0]*k
				for j in range(k):
					temp[j] = mvn.pdf(data[i],mean[j],sigma[j])*phi[j]
				denom = sum(temp)
				# print("denom",denom)
				for j in range(k):	
					w_s[i][j] = temp[j]/denom

			# M-STEP
			for j in range(k):

				sum_temp = 0
				for i in range(len(data)):
					sum_temp = sum_temp + w_s[i][j]
				phi[j] = sum_temp/len(data)

				mean_temp = [0]*len(data[0])
				for i in range(len(data)):
					temp = [0]*len(data[i])
					for t in range(len(temp)):
						temp[t] = w_s[i][j]*data[i][t]
					mean_temp = my_sum_vector(mean_temp,temp)
				for i in range(len(mean[j])):
					mean[j][i] = mean_temp[i]/sum_temp

				sigma_temp = numpy.array([0]*(len(data[0])**2)).reshape(len(data[0]),len(data[0]))
				for i in range(len(data)):
					ar_temp = numpy.array(my_subtract(data[i],mean[j]))
					mat_mul = numpy.matmul(ar_temp.reshape(len(data[0]),1),ar_temp.reshape(1,len(data[0])))
					for t1 in range(len(mat_mul)):
						for t2 in range(len(mat_mul[0])):
							mat_mul[t1][t2] = w_s[i][j]*mat_mul[t1][t2]
					sigma_temp = my_sum_matrix(sigma_temp,mat_mul)
				for i1 in range(len(sigma[j])):
					for i2 in range(len(sigma[j][0])):
						sigma[j][i1][i2] = sigma_temp[i1][i2]/sum_temp				

		sop_ = sum_of_probs(k,phi,mean,sigma)
		
		print(k)
		print(phi)
		print(mean)
		print(sigma)
		print(sop)

		if sop_ > sop:
			k_final = k
			sop = sop_
			phi_final = phi
			mean_final = mean
			sigma_final = sigma

	return k_final,phi_final,mean_final,sigma_final


# Replace this by some parameter directly proportional to betterness of k
def sum_of_probs(k,phi,mean,sigma):
	sop = 0
	for point in data:
		prediction = predict(k,phi,mean,sigma,point)
		sop = sop + prediction
	return sop

def predict(k,phi,mean,sigma,point):
	ans = 0
	for i in range(k):
		ans = ans + mvn.pdf(point,mean[i],sigma[i])*phi[i]
	return ans

# To decide best value of k, choose a value that gives highest sum of probabilities


# data,_,_ = get_data("data_ul.csv")

# only for 2-D data
def check_de_output():
	with open('data_predicted.csv','a') as file:
		for x in range(-20,21):
			for y in range(-20,21):
				prediction = predict(k,phi,mean,sigma,[x,y])
				# print(str(x) + " " + str(y) + " " + str(prediction))
				file.write(str(x) + " " + str(y) + " " + str(prediction) + "\n")

k,phi,mean,sigma = train()
print(phi)
print(mean)
print(sigma)
print(k)
# check_de_output()