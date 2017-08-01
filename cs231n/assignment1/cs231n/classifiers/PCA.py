def PCA(train, val, test, output_dim):
	n, dim = train.shape
	cov_matrix = np.dot(train.T, train)/n
	w, v = np.linalg.eig(cov_matrix)
	new_train = train.dot(w[:, :output_dim])
	new_val = val.dot(w[:, :output_dim])
	new_test = test.dot(w[:, :output_dim])
	return new_test, new_val, new_test