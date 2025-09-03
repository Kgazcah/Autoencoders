from preprocessing.interface_builder import Builder

class Director(object):
  def __init__(self):
    self.builder=None 

  def makeBasicPreprocessing(self, builder):
      self.builder=builder
      self.builder.reset()
      self.builder.build_lowercase_text('text')
      self.builder.build_remove_punctuation()
      self.builder.build_lemmatize()
      return self.builder.getProduct()
  
  def makePlusPreprocessing(self, builder):
      self.builder=builder
      self.builder.reset()
      self.builder.build_lowercase_text('text')
      self.builder.build_remove_punctuation()
      self.builder.build_removing_stopwords()
      self.builder.build_lemmatize()
      return self.builder.getProduct()

  
