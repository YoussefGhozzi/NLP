import spacy
from spacy.util import get_package_path

model_path = get_package_path('fr_core_news_sm')
print(model_path)
