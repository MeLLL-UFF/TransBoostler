
from ekphrasis.classes.segmenter import Segmenter
from abbreviations import abbreviations
from nltk.stem import WordNetLemmatizer
#from nltk.stem import PorterStemmer
from nltk import pos_tag

class Preprocessing:

	def __init__(self, segmenter):
		# Segmenter using the word statistics from Wikipedia
		self.seg = segmenter
		#self.porter_stemmer = PorterStemmer()
		self.wordnet_lemmatizer = WordNetLemmatizer()

	def __check_for_abbreviations(self, text):
		"""
			Replace abbreviations using a dictionary

			Args:
				text(list): list of words in a predicate
			Returns:
				list of the same words but abbreviations replaced by the full word
		"""
		full_words = []
		for word in text:
			if word in abbreviations:
				full_words += abbreviations[word].split()
			else:
				full_words.append(word)
		return full_words

	def __remove_special_characters_from_list(literals):
		"""
			Remove special characters from string

			Args:
				literals(list): list of predicates/literals
			Returns:
				list of the same predicates/literals with no special characters
		"""
		for i in range(len(literals)):
			literals[i] = remove_special_characters(literals[i])
		return atoms

	def __remove_special_characters(string):
		"""
			Remove special characters from string

			Args:
				string(array): predicate/literal
			Returns:
				string with no special characters
		""" 
		return re.sub('[^A-Za-z0-9]+', '', string)

	def __segment(self, text):
		"""
			Remove special characters from string

			Args:
				atoms(list): list of predicates/literals
			Returns:
				list of the same predicates/literals with no special characters
		"""

		# Tokenize (segment) the predicates into words
		# wasbornin -> was, born, in
		return self.seg.segment(text)

	def pre_process_text(self, text):
		"""
			Data preprocessing

			Args:
				text(str): a single predicate
			Returns:
				the predicate after lemma
		"""
		predicate = []
		for word_tag in pos_tag(self.__check_for_abbreviations(self.__segment(text).split())):
			# If it's a verb, we apply lemmatization
			predicate.append(self.wordnet_lemmatizer.lemmatize(word_tag[0], pos="v"))
		return predicate

#test = Preprocessing()
#print(test.pre_process_text('actor'))