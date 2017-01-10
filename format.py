import scipy.io

mat = scipy.io.loadmat("train_32x32.mat")

inputs = mat["X"]
outputs = mat["y"]

out_file = open("dataset_train.csv", "w")

for i in xrange(0, len(inputs[0][0][0])):
	if i % 100 == 0:
		print i

	out_file.write(str(outputs[i][0]))
	for x in xrange(0, 32):
		for y in xrange(0, 32):
			grey = ((int(inputs[x][y][0][i]) + int(inputs[x][y][1][i]) + int(inputs[x][y][2][i])) / 3.0) / 255
			out_file.write("," + str(grey))

	out_file.write("\n")