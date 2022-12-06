import glob

def extract_labels(dir_path):
	splittedby_ = dir_path.split("_")
	splittedbyslash = splittedby_[-2].split("/")
	label = splittedbyslash[-2]
	gripPower = splittedbyslash[-1]
	weight = int(splittedby_[-1])

	return label, gripPower, weight

def read_data():
	dir_paths = glob.glob("./Visual-Tactile_Dataset/dataset/*/*")

	ideal_paths = []

	temp_arr = []
	last_label, _, _ = extract_labels(dir_paths[0])
	final_label, _, _ = extract_labels(dir_paths[-1])

	for dir_path in dir_paths:
		label, gripPower, weight = extract_labels(dir_path)
		if(gripPower == "50"):
			if(label != last_label) or (label == final_label and len(temp_arr) > 1):
				max_idx = 0

				for idx, p in enumerate(temp_arr):
					if(p[2] > temp_arr[max_idx][2]):
						max_idx = idx
				
				ideal_paths.append((temp_arr[max_idx][-1], temp_arr[max_idx][0]))
				temp_arr = []
				last_label = label

			temp_arr.append((label, gripPower, weight, dir_path))

	data = []
	labels = []

	for path, label in ideal_paths:
		paths = glob.glob(path + "/*")
		paths = glob.glob(paths[1] + "/*/pos.txt") # grab top poses

		for p in paths:
			with open(p) as f:
				x = []
				lines = f.readlines()
				for l in lines:
					x.append([float(num) for num in l.split(" ")])
				data.append(x)

		labels.append(label)

	return data, labels