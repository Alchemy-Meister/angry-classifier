# -*- coding: utf-8 -*-

import json, arff
import re

# Open Dataset and load into a JSON.
json_file = open('./../datasets/SMILE/smile-annotations-final.json')
json_data = json.load(json_file)

# Define and initialize arff data dictionary.
data_dict = {'attributes': [], 'data': [], 'description': '', 'relation': ''}
data_dict['relation'] = 'Twitter-Messages'
data_dict['attributes'] = [('text', 'STRING'), ('class', ['angry', 'disgust', \
	'happy', 'sad', 'surprise', 'nocode', 'not-relevant'])]

for data in json_data:
	try:
		# Twitter preprocessing: replacing URLs and Mentions
		twitter_url_str = (ur'http[s]?://(?:[a-zA-Z]|[0-9]|[$+*%/@.&+]' \
			ur'|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

		twitter_mention_regex = re.compile(ur'(\s+|^|.)@\S+')
		twitter_url_regex = re.compile(twitter_url_str)

		# Clean text of new lines.
		preprocessed_tweet = data['text'].replace('\n', ' ');

		# Remove urls and mentions with representative key code.
		preprocessed_tweet = re.sub(twitter_mention_regex, ' USER', \
			preprocessed_tweet)
		preprocessed_tweet = re.sub(twitter_url_regex, 'URL', \
			preprocessed_tweet)

		# Trims generated string.
		preprocessed_tweet = preprocessed_tweet.strip()

		# Load arff dictionary data with dataset data.
		data_dict['data'].append([preprocessed_tweet, data['emotions'][0]])
	except:
		pass

try:
	# Generate arff format string.
	arff_data = arff.dumps(data_dict)
	
	# Write arff string into file.
	arff_output = open('./../datasets/SMILE/smile.arff', 'w+')
	arff_output.write(arff_data.encode('utf8'))
	arff_output.close()

except:
	pass