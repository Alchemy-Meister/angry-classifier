# /usr/bin/env
# -*- coding: utf-8 -*-

import arff
import csv
import re

# Open CSV file and load its data.
csv_file = open('./../datasets/crowdflower/text_emotion.csv', 'rb')
csv_data = csv.reader(csv_file)

# Define and initialize arff data dictionary.
data_dict = {'attributes': [], 'data': [], 'description': '', 'relation': ''}
data_dict['relation'] = 'Twitter-Messages'
data_dict['attributes'] = [('text', 'STRING'), ('@@class@@', ['anger', \
	'boredom', 'enthusiasm', 'empty', 'fun', 'happiness', 'hate', 'love', \
	'neutral', 'relief', 'sadness', 'surprise', 'worry'])]

header_row = True

for row in csv_data:
	if header_row:
		# Do noathing with the header.
		header_row = False
	else:
		try:
			# Twitter preprocessing: replacing URLs and Mentions
			twitter_url_str = (ur'http[s]?://(?:[a-zA-Z]|[0-9]|[$+*%/@.&+]' \
				ur'|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

			twitter_mention_regex = re.compile(ur'(\s+|^|.)@\S+')
			twitter_url_regex = re.compile(twitter_url_str)

			# Clean text of new lines.
			preprocessed_tweet = row[3].replace('\n', ' ');

			# Remove urls and mentions with representative key code.
			preprocessed_tweet = re.sub(twitter_mention_regex, ' AT_USER', \
				preprocessed_tweet)
			preprocessed_tweet = re.sub(twitter_url_regex, 'URL', \
				preprocessed_tweet)

			# Trims generated string.
			preprocessed_tweet = preprocessed_tweet.strip()

			# Escape \ character.
			preprocessed_tweet = preprocessed_tweet.replace('\\', '\\\\')

			# Load arff dictionary data with dataset data.
			data_dict['data'].append([preprocessed_tweet.encode('utf8'), \
				row[1]])

		except:
			pass

try:
	# Generate arff format string.
	arff_data = arff.dumps(data_dict)
	
	# Write arff string into file.
	arff_output = open('./../datasets/crowdflower/crowdflower.arff', 'w+')
	arff_output.write(arff_data)
	arff_output.close()

except:
	pass