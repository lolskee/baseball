"""
Runs predictions on a DraftKings salaries csv for a given mlb slate. 

Use run_dk_predictions.sh to implement 
"""
__author__ = 'lolskee'

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np 
import argparse
import os 
import pickle

# BATTING_TRAIN_COLS = pickle.load(open('../pickles/batter_train_columns.pkl'))
# PITCHER_TRAIN_COLS = pickle.load(open('../pickles/pitcher_train_columns.pkl'))

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dk_salaries")
	parser.add_argument("--date")
	parser.add_argument("--data_dir")
	parser.add_argument("--hitters_model")
	parser.add_argument("--pitchers_model")
	return parser.parse_args()

class DataError(Exception):
	pass

class Batter:
	def __init__(self, dk_player):
		self.name = dk_player.Name 
		self.position = dk_player.Position 
		self.salary = dk_player.Salary/1e3
		self.game_info = dk_player.GameInfo 

	def match_opposing_pitcher(self, lineups):
		#TODO match up batter with the pitcher he's going against 
		pass

	def fetch_batter_stats_and_splits(self, data_dir):
		#TODO retrieve batting stats and appropriate splits based on pitcher
		pass

	def fetch_vegas_lines_and_park_factors(self, vegas_df):
		#TODO match up home team and ballpark
		#using ../mlb_data/park_factors/espn2016_park_factors.csv
		pass

	def to_dict(self):
		return {a: getattr(self, a) for a in vars(a)}

def get_and_parse_daily_lineups(slate_date):
	#TODO figure out whats going on with forecast.io links - they're not working 
	br_url = 'http://www.baseballpress.com/lineups/{}'.format(slate_date)
	soup = BeautifulSoup(urlopen(br_url).read(), "lxml")
	game_divs = soup.findAll('div', {'class': 'game'})
	games = dict()
	weather_divs = soup.findAll('div', {'class': 'weather'})
	for game_div, weather_div in zip(game_divs, weather_divs):
		game_data = str()
		for a in game_div.findAll('a'):
			if '/' in a['href']:
				game_data += a['href'].split('/')[-1] + ','
			else:
				game_data += a.text + ','
		game_data_list = game_data.split(',')
		game = game_data_list[0] + '@' + game_data_list[2]
		games[game] = {
			'away_SP': game_data_list[1],
			'home_SP': game_data_list[3],
			'away_lineup': game_data_list[5:14],
			'home_lineup': game_data_list[15:24],
			'weather_forecast': weather_div.a['href']
		}
	return games 

def scrape_mlb_vegas_lines(slate_date):
	money_line_url = 'http://www.sportsbookreview.com/betting-odds/mlb-baseball/?date={}'
	over_under_url = 'http://www.sportsbookreview.com/betting-odds/mlb-baseball/totals/?date={}'
	sbr_date = slate_date.replace('-', '')

	#scrape money lines 
	# print(sbr_date)
	u = money_line_url.format(sbr_date)
	# print(u)
	ml_soup = BeautifulSoup(urlopen(u).read(), "lxml")
	game_data_list = list()
	game_data_header = 'month_day,home_team,away_team,home_pitcher,away_pitcher,home_ml,away_ml'
	month_day = sbr_date[-4:] if sbr_date[-4:][0] != '0' else sbr_date[-3:]
	for div in ml_soup.findAll('div', {'class': 'data'}):
		for game in div.findAll('div', {'class': 'event-holder'}):
			try:
				for i, tag in enumerate(game.findAll()):
					# print(i, tag.text)
					if i == 27:
						# print(i, tag.text)
						away_team, away_pitcher = tag.text.split('-')
					if i == 32:
						# print(i, tag.text)
						home_team, home_pitcher = tag.text.split('-')
					if i == 49:
						# print(i, tag.text)
						away_ml = tag.text
					if i == 51:
						# print(i, tag.text)
						home_ml = tag.text
				gd = [month_day, home_team, away_team, 
					home_pitcher, away_pitcher, home_ml, away_ml[-4:]]
				#get rid of non ascii chars
				gd = [''.join([i if ord(i) < 128 else '' for i in text]) for text in gd]
				game_data_list.append(gd)
			except Exception as e:
				#will throw error if game is postponed
				print(e)
				pass

	#scrape over/unders
	ou_soup = BeautifulSoup(urlopen(over_under_url.format(sbr_date)).read(), "lxml")
	ou_data_list = list()
	for div in ou_soup.findAll('div', {'class': 'data'}):
		for game_index, game in enumerate(div.findAll('div', {'class': 'event-holder'})):
			try:
				for i, tag in enumerate(game.findAll()):
					# print(i, tag.text)
					if i == 27:
						ou_away_pitcher = tag.text.split('-')[1]
					if i == 45:
						over_under_text = tag.text.replace('+', '-').split('\xa0')[0]
						#deal with weird chars when over/under is a fraction
						try:
							over_under = float(over_under_text)
						except:
							over_under = float(over_under_text[:-1]) + 0.5
				ou_data_list.append([ou_away_pitcher, over_under])
			except Exception as e:
				print(e)
				pass

	game_data_df = pd.DataFrame(game_data_list, columns=game_data_header.split(','))
	ou_df = pd.DataFrame(ou_data_list, columns=['away_pitcher', 'over_under'])
	ou_df.away_pitcher = ou_df.away_pitcher.apply(lambda x: 
		''.join([i if ord(i) < 128 else '' for i in x]))

	return game_data_df.merge(ou_df, on='away_pitcher', how='left')

def main():
	"""
	Makes DK point predictions for each player in dk_salaries. 
	"""
	#parse cmd line args 
	args = parse_args()

	#read in DKSalaries csv and split into pitchers and batters 
	salaries_df = pd.read_csv(args.dk_salaries)
	pitcher_df = salaries_df[salaries_df.Position == 'SP']
	batter_df = salaries_df[salaries_df.Position != 'SP']

	#get daily lineups and vegas lines
	lineups = get_and_parse_daily_lineups(args.slate_date)
	vegas_lines = get_and_parse_vegas_lines(args.slate_date)
	
	#get batter data 
	Batters = [Batter(dk_row) for _, dk_row in batter_df.iterrows()]
	for b in Batters:
		try:
			b.match_opposing_pitcher(lineups)
			b.fetch_batter_stats_and_splits(args.data_dir)
			b.fetch_vegas_lines_and_park_factors(vegas_lines)
		except Exception as e:
			raise DataError('Error occured for batter "{}" due to {}'.format(
				b.name, e))
	
	batter_dicts = [b.to_dict() for b in Batters]
	batter_data_df = pd.DataFrame(batter_dicts)

	#make batter predictions 
	batter_model = joblib.load(args.hitters_model)
	batter_predictions = batter_model.predict(batter_data_df[BATTING_TRAIN_COLS])
	batter_df['prediction'] = batter_predictions

	#get pitcher data
	pitcher_data_df = master[pitcher_cols]
	pitcher_data_df_agg = pitcher_df.groupby(['pitcher'], as_index=False).mean()
	pitcher_data_df_agg = pitcher_df_agg[pitcher_df_agg.over_under >= 6]

	#make pitcher predictions
	pitcher_model = joblib.load(args.pitchers_model)
	pitcher_predictions = pitchers_model.predict(pitcher_data_df_agg[PITCHER_TRAIN_COLS])
	pitcher_df['prediction'] = pitcher_predictions

	#combine pitchers and batters and export predictions to csv
	slate_df = pd.concat([pitcher_df, batter_df], axis=1)
	slate_df.to_csv('dk_predictions/{}.csv'.format(args.slate_date), index=False)

if __name__ == '__main__':
	main()


