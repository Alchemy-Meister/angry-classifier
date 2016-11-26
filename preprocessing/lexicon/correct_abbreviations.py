import codecs
import json
import collections

dictionary = collections.OrderedDict()
di = {}

file = open('./twitter_abbreviations.json', 'r')
integer = 0
for line in file:
	if integer != 0 and integer != 1377:
		key_value = line.split(':')
		if len(key_value) == 3:
			key = (key_value[0] + ':' + key_value[1]).replace('"', '')
			value = key_value[2][:-2]
			value = value.replace('[', '')
			value = value.replace(']', '')
			values = value.split('",')
			for value in values:
				value = value.strip()
				value = value.replace('"', '')
				value = value.lower()
		else:
			key = key_value[0].replace('"', '')
			value = key_value[1][:-2]
			value = value.replace('[', '')
			value = value.replace(']', '')
			values = value.split('",')
			for value in values:
				value = value.strip()
				value = value.replace('"', '')
				value = value.lower()
		key = key.strip()
		if key not in dictionary:
			dictionary[key] = []

		dictionary[key].append(value)

	integer += 1

for key, value in dictionary.items():
	di[key] = value

with codecs.open('./corrected_twitter_abbreviations.json', 'w', encoding='utf-8') as ofile:
	ofile.write(json.dumps(di, indent=4, sort_keys=True))