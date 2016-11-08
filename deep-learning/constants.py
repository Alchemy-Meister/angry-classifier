# -*-*- encoding: utf8 -*-*-
JSON_RAW_DIR = "./raw_json/"
MINI_JSON_RAW_DIR = "./minified_raw_json/"
PIECES_JSON_DIR = "./pieces_json/"
MINI_PIECES_JSON_DIR = "./minified_pieces_json/"
PREVECTOR_JSON_DIR = "./prevector_json/"
MINI_PREVECTOR_JSON_DIR = "./minified_prevector_json/"
POSTVECTOR_JSON_DIR = "./postvector_json/"
MINI_POSTVECTOR_JSON_DIR = "./minified_postvector_json/"
PADDED_JSON_DIR = "./padded_json/"
MINI_PADDED_JSON_DIR = "./minified_padded_json/"
SAVE_STATS_DIR = "./computed_data/"
MINI_SAVE_STATS_DIR = "./minified_computed_data/"
SEPARATION_CHARS = [u'.', u'-', u'/', u'|', u'_', u'\\', u'"', u'(', u')', u',', u';', u':', u'[', u']', u'!', u'¡', u'¿', u'?', u'=',u'&', u'º', u'ª']
MODELS_DIR = "./models/"
MODEL_NAME = "complete_without_stopwords"

PATTERN = '^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
MAX_PHRASE_LENGTH = 60
SPLIT_INTO_PIECES = True
NUM_SPLITS = 2

SPLIT_SENTENCES = False

WEIRD_CHARS_CODES = {
    u'\u000b': u'',
    u'\u001f': u'',
    u'\r': u'',
    u'\r\x07': u'',
    u'\u0007': u'',
    u' ': u'',
    u'a': u'',
    u'b': u'',
}
# Convert weird characters to its basic form
KNOWN_CHARS = {
    u'\u2010': u'-',
    u'\u2013': u'-',
    u'\u2019': u"'",
    u'\u2018': u"'",
    u'\u201d': u'"',
    u'\u201c': u'"',
    u'\u2026': u'...',
    u'\u20ac': u' euros ',
    u'\xb4' : u"'",
    u'\xa1': u'¡',
    u'\xa3': u'£',
    u'\xa4': u' euros ',
    u'\ufb01': u'fi',
    u'\ufb02': u'fl',
    u'\u2011': u'-',
    u'\u2212': u'-',
    u'\u2014': u'-',
    u'\u2015': u'-',
    u'\u201f': u'"',
    u'\u201e': u'"',
    u'\u2122': u'™',
    u'\xa8': u'"',
    u'\xab': u'"',
    u'\xbb': u'"',
    u'\xb0': u'º',
    u'\xb3': u'³',
    u'\xf2': u'ó',
    u'\xbd': u' media ',
    u'\xd6': u'Ó',
    u'\xe0': u'à',
    u'\xe2': u'â',
    u'\u2264': u'≤',
    u'\xe8': u'è',
    u'\xef': u'ï',
    u'\xbc': u'%',
    u'\u2022': u'', # bullet point
    u'\u203a': u'', # ›
    u'\uf08a': u'', # 
    u'\u25a0': u'', # ■
    u'\uf0a7': u'', # 
    u'\xad': u'', # (empty)
    u'\xac': u'', # ¬
    u'\uf02f': u'', #
    u'\xae': u'', # (copyright symbol used as bullet point)
    u'\xb7': u'', # bullet point
    u'\u223c': u'', # ~ used as bullet point
    u'\uf020': u'', # small .
    u'\xa0': u'',
    u'\xa7': u'', # §
    u'\u2192': u'', # →'
    u'\u0e4f': u'', # ๏
    u'\xd4': u'', # Ô as bullet point
    u'\uf0d8': u'', # 
    u'\u2666': u'', # ♦
    u'\u0192': u'', # ƒ
    u'\uf076': u'', # 
    u'\uf095': u'', # 
    u'\uf0fc': u'', # ,
    u'\r\x07': u'',
}
