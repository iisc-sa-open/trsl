class NGramTable():
	"""
	"""

	def __init__(self,arr,n):

		self.array = arr
		self.ngram_window_size = n

	def __len__(self):

		return len(self.array) - (self.ngram_window_size - 1)

	def __getitem__(self,tup):

		row, col = tup
		if row < (len(self.array) - (self.ngram_window_size - 1)) and col < self.ngram_window_size:
			return self.array[row + col]
		else:
			raise KeyError

	def __setitem__(self,tup,item):

		row, col = tup
		if row < (len(self.array) - (self.ngram_window_size - 1)) and col < self.ngram_window_size:
			self.array[row + col] = item
		else:
			raise KeyError
