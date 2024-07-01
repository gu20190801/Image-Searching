import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from flask import Flask, render_template, flash, request, redirect, url_for
from glob import glob
import re
from werkzeug.utils import secure_filename

from sklearn.metrics import precision_score

from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk


base_path = os.getcwd()
app = Flask(__name__, template_folder = 'Template')
upload_folder = 'static/uploads/'
app.secret_key = 'caircoders-ednalan'
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024

############################################# Features Extractions ############################################

def divideImg(img):
	(h, w) = img.shape[:2]
	(newX, newY) = (w//8, h//8)
	newWidth = w - newX
	newHeight = h - newY

	cropped = img[newY:newHeight, newX:newWidth]
	cropped = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
	return cropped

def crop_regions(img, num):
	img = cv.resize(img, (32,32))
	(R1x, R1y) = (8, 8)
	(R2x, R2y) = (16, 16)
	(R3x, R3y) = (32, 32)
	
	if(num == 1):
		cropped = img[0:R1y, 0:R1x]
	if(num == 2):
		cropped = img[0:R2y, 0:R2x]
	if(num == 3):
		cropped = img[0:R3y, 0:R3x]

	return cropped

def findMean(img):
	mean_h = np.average(img[:,:,0])
	mean_s = np.average(img[:,:,1])
	mean_v = np.average(img[:,:,2])

	mean_arr = np.array([mean_h, mean_s, mean_v])
	flattened = mean_arr.flatten()

	return (flattened)


def findSD(img):
	sd_h = np.std(img[:,:,0])
	sd_s = np.std(img[:,:,1])
	sd_v = np.std(img[:,:,2])

	std_arr = np.array([sd_h, sd_s, sd_v])
	flattened = std_arr.flatten()

	return (flattened)


def findEntropy(img):
	entropy_h = entropy(img[:,:,0], disk(5))
	entropy_s = entropy(img[:,:,1], disk(5))
	entropy_v = entropy(img[:,:,2], disk(5))
	
	entropy_arr = np.array([entropy_h,entropy_s,entropy_v])
	flattened = entropy_arr.flatten()

	return flattened


def GLCM(img):
   gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   
   table8 = np.array( [i//16 for i in range(256)] ).astype("uint8")  #16 level compression
   gray8 = cv.LUT(gray, table8)

   dist = [1,2]
   degree = [0, np.pi/4, np.pi/2, np.pi*3/4]
   glcm = graycomatrix(gray8, dist, degree, levels = 16)
   
   feature_cont = graycoprops(glcm,'contrast')
   feature_corr = graycoprops(glcm,'correlation')
   feature_energy = graycoprops(glcm,'energy')
   feature_homo = graycoprops(glcm,'homogeneity')
   
   result = np.array([feature_cont,feature_corr,feature_energy,feature_homo])
   flattened = result.flatten()
   
   return flattened

def feature_extraction(img):
	mean_h = np.array(findMean(crop_regions(divideImg(img), 1))).flatten()
	mean_s = np.array(findMean(crop_regions(divideImg(img), 2))).flatten()
	mean_v = np.array(findMean(crop_regions(divideImg(img), 3))).flatten()

	std_h = np.array(findSD(crop_regions(divideImg(img), 1))).flatten()
	std_s = np.array(findSD(crop_regions(divideImg(img), 2))).flatten()
	std_v = np.array(findSD(crop_regions(divideImg(img), 3))).flatten()

	entropy_h = np.array(findEntropy(crop_regions(divideImg(img), 1))).flatten()
	entropy_s = np.array(findEntropy(crop_regions(divideImg(img), 2))).flatten()
	entropy_v = np.array(findEntropy(crop_regions(divideImg(img), 3))).flatten()

	glcm_h = np.array(GLCM(crop_regions(divideImg(img), 1))).flatten()
	glcm_s = np.array(GLCM(crop_regions(divideImg(img), 2))).flatten()
	glcm_v = np.array(GLCM(crop_regions(divideImg(img), 3))).flatten()
	
	data = np.hstack((mean_h, mean_s, mean_v, std_h, std_s, std_v, entropy_h, entropy_s, entropy_v, glcm_h, glcm_s, glcm_v)).flatten()

	return data

############################################# Save Image Dataset ############################################

def save_image_dataset():
	dir = "image_dataset"

	categories = ['Beach', 'Building', 'Bus', 'Dinosaur', 'Flower', 'Horse', 'Man', 'Elephants', 'Mountains', 'Foods']

	data = []

	for category in categories:
		path = os.path.join(dir, category)
		label = categories.index(category)

		for img in os.listdir(path):
			imgpath = os.path.join(path, img)
			train_img = cv.imread(imgpath)
			try:
				train_img = cv.resize(train_img, (64, 64))
				extracted_features = np.array(feature_extraction(train_img)).flatten() #39D features

				data.append([extracted_features, label])
			except Exception as e:
				pass

	print(len(data))

	pick_in = open('data.pickle', 'wb')
	pickle.dump(data, pick_in)
	pick_in.close()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
	return render_template('index.html')

################################################ Retrieval #########################################

@app.route('/', methods=['POST'])
def upload_file():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No selected file')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded and displayed below')
		return redirect(url_for('retrieval', filename=filename))
	else:
		flash('Allowed image types are - png, jpg, jpeg, gif')
		return redirect(request.url)
	
	return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/'+filename), code=301)

@app.route('/<filename>')
def retrieval(filename):
	################################################ Read Image Dataset #########################################
	count=0
	global base_path
	global upload_folder
	os.chdir(base_path)
	pick_in = open('data.pickle', 'rb')
	data = pickle.load(pick_in)
	pick_in.close()

	random.shuffle(data)
	features = []
	labels = []

	for feature, label in data:
		features.append(feature)
		labels.append(label)

	xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size = 0.25,  random_state=0)

	################################################ Train Model #########################################

#	model = SVC(C = 1, kernel = 'poly', gamma = 'auto', probability = True)
#	model.fit(xtrain, ytrain)

#	pick = open('model.sav', 'wb')
#	pickle.dump(model, pick)
#	pick.close()

	################################################### Prediction Section ######################################
	pick = open('model.sav', 'rb')
	model = pickle.load(pick)
	pick.close()

	categories = ['Beach', 'Building', 'Bus', 'Dinosaur', 'Flower', 'Horse', 'Man', 'Elephants', 'Mountains', 'Foods']

	accuracy = model.score(xtest, ytest)
	print ('Accuracy: ', accuracy.round(2))


	######### Query Image #########
	file_path = os.path.join(base_path, upload_folder)
	file_path = os.path.join(file_path, filename)
	print(file_path)
	src_input = cv.imread(file_path)
	query_img = src_input
	query_features = [np.array(feature_extraction(query_img)).flatten()]

	query_prediction = model.predict(query_features)

	query_category = categories[query_prediction[0]]
	indicator = int(100*((query_prediction[0]+1)%7))


	print ('input success')

	static = "static"

	static_path = os.path.join(base_path, static)

	isExist = os.path.exists(static_path)
	if not isExist:
		os.mkdir(static_path)

	# the directory of the image database
	database_dir = os.path.join(base_path, "image.orig")
	# read image database
	database = sorted(glob(database_dir + "/*.jpg"))

	positive_images = []
	true_positive_images = []
	false_negative_images = []
	print(base_path)
	for img in database:
		# read image
		os.chdir(os.path.join(base_path, database_dir))
		candidate_img = cv.imread(img)

		######### Candidate Image #########
		candidate_features = [np.array(feature_extraction(candidate_img)).flatten()]

		####### Proba #######
		best = ""
		max = 0
		probability = model.predict_proba(candidate_features)
		for ind,val in enumerate(categories):
			if(probability[0][ind]*100 > max):
				max = probability[0][ind]*100
				best = probability[0][ind]*100
	
		########### Predict ##########
		candidate_prediction = model.predict(candidate_features)
		candidate_category = categories[candidate_prediction[0]]

		os.chdir(database_dir)
		########### Count Same Category Prediction ###########
		img_number = re.findall(r'\d+', img)
		print(img_number)
		img_number = int(img_number[1])
		if ( img_number >= indicator and img_number <= indicator + 99 ):
			if(query_category != candidate_category):
				false_negative_images.append([candidate_img, best])

		########### Count Same Category Prediction ###########
		if(query_category == candidate_category):
			positive_images.append([candidate_img, best])
			if ( img_number >= indicator and img_number <= indicator + 99 ):
				true_positive_images.append([candidate_img, best])

		print (candidate_category, ' ', best.round(2), "%")


	############## Precision Rate and Recall Rate ##################

	precision_rate = len(true_positive_images)/len(positive_images)*100
	recall_rate = len(true_positive_images)/(len(false_negative_images)+len(true_positive_images))*100
	print("Precision Rate: ", round(precision_rate, 2), "%" )
	print("Recall Rate: ", round(recall_rate, 2), "%" )
	
	############## Print Multiple Result by result[] ##################
	# find 25 best images
	result = []
	result_value = []
	sorted_positive_images = sorted(positive_images, key=lambda x: x[1], reverse=True)
	for i in range(25):
		result.append(sorted_positive_images[i][0])
		result_value.append(str(sorted_positive_images[i][1].round(2))+"%")
	directory = query_category

	path = os.path.join(static_path, directory)
	isExist = os.path.exists(path)
	if not isExist:
		os.mkdir(path)

	print ('Folders are built')

	for k in range(25):
		img = result[k]
		
		write_path = os.path.join(static_path, query_category)

		os.chdir(write_path)
		img = np.array(img)
		img_name = count
		cv.imwrite(str(img_name)+'.jpg', img)
		count += 1

	list_path = os.path.join(static_path, query_category)
	imageList = os.listdir(list_path)
	imagelist = [query_category +'/' + image for image in imageList]
	return render_template("index.html", imagelist = imagelist, filename = filename, valuelist = result_value, category = query_category)

if __name__ == '__main__':
      app.run(host='127.0.0.1', port=8000)
