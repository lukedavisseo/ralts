import streamlit as st
import pandas as pd
import json
from transformers import pipeline
import textrazor
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import numpy as np
import random

# TextRazor details
textrazor.api_key = 'API_KEY'

client = textrazor.TextRazor(extractors=["entities", "topics"])
client.set_classifiers(["textrazor_newscodes"])

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
	return pipeline("zero-shot-classification", model='valhalla/distilbart-mnli-12-9', multi_label=True)

# Classification function
def classify(sequences, candidate_labels):
	output_results = load_model()(sequences=sequences, candidate_labels=candidate_labels)
	return output_results['labels'], output_results['scores']

# Graph plot function
def plot_result(top_topics, scores):
	top_topics = np.array(top_topics)
	scores = np.array(scores)
	scores *= 100
	fig = px.bar(x=scores, y=top_topics, orientation='h', 
				 labels={'x': 'Confidence', 'y': 'Label'},
				 text=scores,
				 range_x=(0,115),
				 title='Top Predictions',
				 color=np.linspace(0,1,len(scores)),
				 color_continuous_scale='GnBu')
	fig.update(layout_coloraxis_showscale=False)
	fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
	st.plotly_chart(fig)

def req(url):
	resp = requests.get(url)
	soup = BeautifulSoup(resp.content, 'html.parser')
	if soup.find("div", id="comments") and soup.find("div", id="secondary"):
		remove_comments = soup.find("div", id="comments")
		remove_comments.extract()
		remove_secondary = soup.find("div", id="secondary")
		remove_secondary.extract()
		ext_t = [t.text for t in soup.find_all(['h1', 'p'])]
		paragraphs = ' '.join(ext_t)
		return paragraphs
	elif soup.find("div", id="comments") and soup.find("aside", id="secondary"):
		remove_comments = soup.find("div", id="comments")
		remove_comments.extract()
		remove_secondary = soup.find("aside", id="secondary")
		remove_secondary.extract()
		ext_t = [t.text for t in soup.find_all(['h1', 'p'])]
		paragraphs = ' '.join(ext_t)
		return paragraphs
	else:
		ext_t = [t.text for t in soup.find_all(['h1', 'p'])]
		paragraphs = ' '.join(ext_t)
		return paragraphs

# Main function
def main():
	with st.spinner('Classifying...'):
		global txt
		df_classify_topic = pd.DataFrame(topics_dict)
		df_classify_topic = df_classify_topic.sort_values(by='Relevance Score', ascending=False)
		classify_topic_score = list(df_classify_topic['Topic'][:10])
		if input_type == 'Text':
			top_topics, scores = classify(txt, classify_topic_score)
		elif input_type == 'URL':
			txt = req(url)
			top_topics, scores = classify(txt, classify_topic_score)
		elif input_type == 'Multiple URLs':
			txt = ' '.join(all_txt)
			top_topics, scores = classify(txt, classify_topic_score)

	plot_result(top_topics[::-1][-10:], scores[::-1][-10:])

# Dictionaries

kw_dict = {
	'Keyword': []
}

ent_dict = {
	'Keyword': [],
	'Relevance Score': [],
	'Existing Tag': []
}

topics_dict = {
	'Topic': [],
	'Relevance Score': [],
	'Existing Tag': []
}

categories_dict = {
	'Category': [],
	'Relevance Score': []
}

tags = {
	'Tag': [],
	'ID': [],
	'Count': []
}

# Blog list
blogs = ['sampleface.co.uk', 'cultrface.co.uk', 'logicface.co.uk', 'playrface.co.uk', 'distantarcade.co.uk']

# Empty lists for existing tags and text from multiple URLs to go in
existing_tags = []
all_txt = []

# Streamlit stuff

st.sidebar.title('Tag suggester')

input_type = st.sidebar.radio('Select your input type', ['Text', 'URL', 'Multiple URLs'])

update_tags = st.sidebar.button('↻ Refresh tags')

tag_ideas = st.sidebar.button('Load tag ideas')


# Determines input types
st.title('Welcome to RALTS (Really Awesome Lexicon and Tag Suggester)!')
st.write('This script can analyse any body of text or URL to find extract keywords, topics, and categories using NLP (natural language processing).')
if input_type == 'Text':
	txt = st.text_area('Enter text to be analysed...')
	txt = txt.replace('\n', ' ').replace('"', '').replace('“','').replace('”', '').replace('‘','').replace('’', '').replace("'s", '').replace(",", '')
	st.write(len(txt))
elif input_type == 'URL':
	url = st.text_input('Enter URL')
elif input_type == 'Multiple URLs':
	multi_url = st.text_area('Enter keywords, 1 per line')

# Upper limits for tag page range
upper_limits = {
	'sampleface.co.uk': 4,
	'cultrface.co.uk': 4,
	'logicface.co.uk': 2,
	'playrface.co.uk': 2,
	'distantarcade.co.uk': 2,
	'ld89.org': 1
}

# Reload all tags function
def update_all_tags():

	with st.spinner('Reloading tags...'):

		for blog in blogs:
		
			# Get tag data
			for pg in range(1, upper_limits[blog]+1):

				tag_url = f'https://{blog}/wp-json/wp/v2/tags?per_page=100&page={pg}'
				r_tag = requests.get(tag_url)
				api_tags = r_tag.json()

				for n in range(0,len(api_tags)):
					tags['Tag'].append(api_tags[n]['name'])
					tags['ID'].append(api_tags[n]['id'])
					tags['Count'].append(api_tags[n]['count'])

				with open(f"{blog}.json", "w") as outfile:
					json.dump(tags, outfile)
			tags['Tag'] = []
			tags['ID'] = []
			tags['Count'] = []

list_of_blogs = st.radio("Select the corresponding blog", blogs)

# Load JSON files
with open(f'{list_of_blogs}.json', 'rb') as f:
	blog_json = json.load(f)
	x = blog_json['Tag']
	for n in x:
		existing_tags.append(n)

submit = st.button('Submit')

if update_tags:
	update_all_tags()

# Tag ideas functions

def sf_words():

	with open("sampleface.co.uk.json") as sf_json_file:
		sf = json.load(sf_json_file)
	sf_words_lists = sf['Tag']
	return sf_words_lists

def cultr_words():

	with open("cultrface.co.uk.json") as cf_json_file:
		cf = json.load(cf_json_file)
	cf_words_lists = cf['Tag']
	return cf_words_lists

def logic_words():

	with open("logicface.co.uk.json") as lf_json_file:
		lf = json.load(lf_json_file)
	lf_words_lists = lf['Tag']
	return lf_words_lists

def playr_words():

	with open("playrface.co.uk.json") as pf_json_file:
		pf = json.load(pf_json_file)
	pf_words_lists = pf['Tag']
	return pf_words_lists

def da_words():

	with open("distantarcade.co.uk.json") as da_json_file:
		da = json.load(da_json_file)
	da_words_lists = da['Tag']
	return da_words_lists

all_words_list = [sf_words(), cultr_words(), logic_words(), playr_words(), da_words()]

def sampleface():

	st.header('Sampleface ideas')

	for sample in range(5):
		with open("sampleface.co.uk.json") as sf_json_file:
			sf = json.load(sf_json_file)
		sf_words_count_lists = sf['Count']
		try:
			x = [n/(n+1) for n in sf_words_count_lists]
		except ZeroDivisionError:
			continue
			# sf_choices = random.choices(sf_words_count_lists, weights=(1/n), k=2)
		sample = random.choices(sf_words(), x, k=2)
		
		st.write('https://google.com/search?q=' + '+'.join(sample).lower().replace(' ', '+').replace('&', ''))
	
def cultrface():

	st.header('Cultrface ideas')

	for sample in range(5):
		with open("cultrface.co.uk.json") as cf_json_file:
			cf = json.load(cf_json_file)
		cf_words_count_lists = cf['Count']
		try:
			x = [n/(n+1) for n in cf_words_count_lists]
		except ZeroDivisionError:
			continue
			# sf_choices = random.choices(sf_words_count_lists, weights=(1/n), k=2)
		sample = random.choices(cultr_words(), x, k=2)
		
		st.write('https://google.com/search?q=' + '+'.join(sample).lower().replace(' ', '+').replace('&', ''))

def logicface():

	st.header('LOGiCFACE ideas')

	for sample in range(5):
		with open("logicface.co.uk.json") as lf_json_file:
			lf = json.load(lf_json_file)
		lf_words_count_lists = lf['Count']
		try:
			x = [n/(n+1) for n in lf_words_count_lists]
		except ZeroDivisionError:
			continue
		sample = random.choices(logic_words(), x, k=2)
		
		st.write('https://google.com/search?q=' + '+'.join(sample).lower().replace(' ', '+').replace('&', ''))

def playrface():

	st.header('Playrface ideas')

	for sample in range(5):
		with open("playrface.co.uk.json") as pf_json_file:
			pf = json.load(pf_json_file)
		pf_words_count_lists = pf['Count']
		try:
			x = [n/(n+1) for n in pf_words_count_lists]
		except ZeroDivisionError:
			continue
		sample = random.choices(playr_words(), x, k=2)
		
		st.write('https://google.com/search?q=' + '+'.join(sample).lower().replace(' ', '+').replace('&', ''))

def distantarcade():

	st.header('Distant Arcade ideas')

	for sample in range(5):
		with open("distantarcade.co.uk.json") as da_json_file:
			da = json.load(da_json_file)
		da_words_count_lists = da['Count']
		try:
			x = [n/(n+1) for n in da_words_count_lists]
		except ZeroDivisionError:
			continue
		sample = random.choices(da_words(), x, k=2)
		
		st.write('https://google.com/search?q=' + '+'.join(sample).lower().replace(' ', '+').replace('&', ''))

def all_blogs():

	sampleface()
	cultrface()
	logicface()
	playrface()
	distantarcade()

# Keyword extraction function to analyse with TextRazor

def textrazor_extraction(input_type):

	if input_type == 'Text':
		global txt
		response = client.analyze(txt)
		for entity in response.entities():
			if entity.relevance_score > 0:
				ent_dict['Keyword'].append(entity.id)
				ent_dict['Relevance Score'].append(entity.relevance_score)
				if entity.id in existing_tags or entity.id.lower() in existing_tags or entity.id.capitalize() in existing_tags or any(existing_tag in entity.id for existing_tag in existing_tags):
					ent_dict['Existing Tag'].append(1)
				else:
					ent_dict['Existing Tag'].append(0)
		
		for topic in response.topics():
			if topic.score > 0.6:
				topics_dict['Topic'].append(topic.label)
				topics_dict['Relevance Score'].append(topic.score)
				if topic.label in existing_tags or topic.label.lower() in existing_tags or topic.label.capitalize() in existing_tags or any(existing_tag in topic.label for existing_tag in existing_tags):
					topics_dict['Existing Tag'].append(1)
				else:
					topics_dict['Existing Tag'].append(0)
		
		for category in response.categories():
			categories_dict['Category'].append(category.label)
			categories_dict['Relevance Score'].append(category.score)

	elif input_type == 'URL':
		txt = req(url)
		response = client.analyze(txt)
		for entity in response.entities():
			if entity.relevance_score > 0:
				ent_dict['Keyword'].append(entity.id)
				ent_dict['Relevance Score'].append(entity.relevance_score)
				if entity.id in existing_tags or entity.id.lower() in existing_tags or entity.id.capitalize() in existing_tags or any(existing_tag in entity.id for existing_tag in existing_tags):
					ent_dict['Existing Tag'].append(1)
				else:
					ent_dict['Existing Tag'].append(0)
		
		for topic in response.topics():
			if topic.score > 0.6:
				topics_dict['Topic'].append(topic.label)
				topics_dict['Relevance Score'].append(topic.score)
				if topic.label in existing_tags or topic.label.lower() in existing_tags or topic.label.capitalize() in existing_tags or any(existing_tag in topic.label for existing_tag in existing_tags):
					topics_dict['Existing Tag'].append(1)
				else:
					topics_dict['Existing Tag'].append(0)
		
		for category in response.categories():
			categories_dict['Category'].append(category.label)
			categories_dict['Relevance Score'].append(category.score)

	elif input_type == 'Multiple URLs':

		for u in urls:
			txt = req(u)
			all_txt.append(txt)
			response = client.analyze(txt)
			for entity in response.entities():
				if entity.relevance_score > 0:
					ent_dict['Keyword'].append(entity.id)
					ent_dict['Relevance Score'].append(entity.relevance_score)
					if entity.id in existing_tags or entity.id.lower() in existing_tags or entity.id.capitalize() in existing_tags or any(existing_tag in entity.id for existing_tag in existing_tags):
						ent_dict['Existing Tag'].append(1)
					else:
						ent_dict['Existing Tag'].append(0)
			
			for topic in response.topics():
				if topic.score > 0.6:
					topics_dict['Topic'].append(topic.label)
					topics_dict['Relevance Score'].append(topic.score)
					if topic.label in existing_tags or topic.label.lower() in existing_tags or topic.label.capitalize() in existing_tags or any(existing_tag in topic.label for existing_tag in existing_tags):
						topics_dict['Existing Tag'].append(1)
					else:
						topics_dict['Existing Tag'].append(0)
			
			for category in response.categories():
				categories_dict['Category'].append(category.label)
				categories_dict['Relevance Score'].append(category.score)

# DataFrames to present above data
def data_viz():

	if input_type == 'Text':

		# df_kw = pd.DataFrame(kw_dict)
		# # grouped_df_kw = df_kw.groupby(['Keyword']).agg({'Relevance Score': ['mean'], 'Existing Tag': ['max']}).round(3)
		# # grouped_df_kw = grouped_df_kw.reset_index()
		# st.header('Keywords')
		# st.dataframe(df_kw)

		df_ent = pd.DataFrame(ent_dict)
		grouped_df_ent = df_ent.groupby(['Keyword']).agg({'Relevance Score': ['mean'], 'Existing Tag': ['max']}).round(3)
		grouped_df_ent = grouped_df_ent.reset_index()
		st.header('Entities')
		st.dataframe(grouped_df_ent)

		df_topic = pd.DataFrame(topics_dict)
		grouped_df_topic = df_topic.groupby(['Topic']).agg({'Relevance Score': ['mean'], 'Existing Tag': ['max']}).round(3)
		grouped_df_topic = grouped_df_topic.reset_index()
		st.header('Topics')
		st.dataframe(grouped_df_topic)

		df_cat = pd.DataFrame(categories_dict)
		grouped_df_cat = df_cat.groupby(['Category']).agg({'Relevance Score': ['mean']}).round(3)
		grouped_df_cat = grouped_df_cat.reset_index()
		st.header('Categories')
		st.dataframe(grouped_df_cat)

	elif input_type == 'URL':

		df_ent = pd.DataFrame(ent_dict)
		grouped_df_ent = df_ent.groupby(['Keyword']).agg({'Relevance Score': ['mean'], 'Existing Tag': ['max']}).round(3)
		grouped_df_ent = grouped_df_ent.reset_index()
		st.header('Entities')
		st.dataframe(grouped_df_ent)

		df_topic = pd.DataFrame(topics_dict)
		grouped_df_topic = df_topic.groupby(['Topic']).agg({'Relevance Score': ['mean'], 'Existing Tag': ['max']}).round(3)
		grouped_df_topic = grouped_df_topic.reset_index()
		st.header('Topics')
		st.dataframe(grouped_df_topic)

		df_cat = pd.DataFrame(categories_dict)
		grouped_df_cat = df_cat.groupby(['Category']).agg({'Relevance Score': ['mean']}).round(3)
		grouped_df_cat = grouped_df_cat.reset_index()
		st.header('Categories')
		st.dataframe(grouped_df_cat)

	elif input_type == 'Multiple URLs':

		df_ent = pd.DataFrame(ent_dict)
		grouped_df_ent = df_ent.groupby(['Keyword', 'Existing Tag']).describe().round(3)
		grouped_df_ent = grouped_df_ent.reset_index()
		st.header('Entities')
		st.dataframe(grouped_df_ent)

		df_topic = pd.DataFrame(topics_dict)
		grouped_df_topic = df_topic.groupby(['Topic', 'Existing Tag']).describe().round(3)
		grouped_df_topic = grouped_df_topic.reset_index()
		st.header('Topics')
		st.dataframe(grouped_df_topic)

		df_cat = pd.DataFrame(categories_dict)
		grouped_df_cat = df_cat.groupby('Category').describe().round(3)
		grouped_df_cat = grouped_df_cat.reset_index()
		st.header('Categories')
		st.dataframe(grouped_df_cat)
		
# Execute functions
if tag_ideas:
	all_blogs()
elif submit and input_type and input_type == 'Text':
	textrazor_extraction('Text')
	data_viz()
	main()
elif submit and input_type == 'URL':
	textrazor_extraction('URL')
	data_viz()
	main()
elif submit and input_type == 'Multiple URLs':
	urls = [line for line in multi_url.split("\n")]
	textrazor_extraction('Multiple URLs')
	data_viz()
