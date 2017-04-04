
# # My Quest To Conquer The World Of Baseball DFS - Part 3
# 
# ## Opening Day MLB 2017 DraftKings Optimization Demo
# 
# #### Tate Campbell - April 3, 2017

# Welcome back! We finally made it to opening day of the 2017 MLB season! 
# I didn't have time to write up an article this week so I thought I'd just walk 
# you through my first attempt at optimizing a lineup. The first and most daunting 
# task will be transforming the DraftKings salary csv into the form of my training data. 
# I still haven't finished the script that will ultimately do this for me, 
# so there's going to be lots of parsing and joins this time through.
# 
# My first task is to gather all of the features used in my models. 

# ## Let's get started

#Loading Libraries 

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
from urllib.request import urlopen
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import os 
import sys
import pickle 
import tabulate
import warnings
from datetime import datetime
from pprint import pprint
from subprocess import check_output
import ast 
warnings.filterwarnings("ignore")

def scrape_roto_grinders(slate_date):
    url = 'https://rotogrinders.com/lineups/mlb?site=draftkings&date={}'.format(slate_date)
    soup = BeautifulSoup(urlopen(url).read(), "lxml")
    ul = soup.findAll('ul', {'class': 'lineup'})
    game_data = dict()
    games = list()
    pitchers = list()
    for li in ul:
        for i, lineup_card in enumerate(li.findAll('li', {'data-role': 'lineup-card'})):
            game = lineup_card['data-away'] + '@' + lineup_card['data-home']
            games.append(game)
            game_data[game] = dict()
        for i, pitcher_div in enumerate(li.findAll('div', {'class': 'pitcher'})):
            if i % 2 == 0:
                ptext = pitcher_div.text.split()
                game_data[games[int(i/2)]]['away_SP'] = ptext[0] + ' ' + ptext[1]
            if i % 2 == 1:
                ptext = pitcher_div.text.split()
                game_data[games[int(i/2)]]['home_SP'] = ptext[0] + ' ' + ptext[1]
            pitchers.append(ptext[0] + ' ' + ptext[1])
        for i, lineup in enumerate(li.findAll('ul', {'class': 'players'})):
            lineup_list = list()
            for player in lineup.findAll('li', {'class': 'player'}):
                player_text = player.text.split()
                lineup_list.append(player_text[1] + ' ' + player_text[2])
            if i % 2 == 0:
                game_data[games[int(i/2)]]['away_lineup'] = lineup_list
            if i % 2 == 1:
                game_data[games[int(i/2)]]['home_lineup'] = lineup_list
    return game_data

def most_recent_dk_salaries(path):
    mtime = lambda f: -os.stat(os.path.join(path, f)).st_mtime
    sorted_files = list(sorted(os.listdir(path), key=mtime))
    for file in sorted_files:
        if 'DKSalaries' in file:
            return path + file

def get_starters(lineups):
    starters = list()
    for game, info in lineups.items():
        starters.append(info['away_SP'])
        starters.append(info['home_SP'])
        if len(info['away_lineup']) > 0:
            starters += info['away_lineup']
        if len(info['home_lineup']) > 0:
            starters += info['home_lineup']
    return starters

def get_pitcher_dict(lineups):
    return {game: [info['away_SP'], info['home_SP']] for game, info in lineups.items()}

def stack_vegas_lines_by_pitcher(vegas_lines):
    home_pitchers = vegas_lines.drop('away_pitcher', axis=1)
    away_pitchers = vegas_lines.drop('home_pitcher', axis=1)
    home_pitchers.rename(index=str, columns={"home_pitcher": "pitcher"}, inplace=1)
    away_pitchers.rename(index=str, columns={"away_pitcher": "pitcher"}, inplace=1)
    return pd.concat([home_pitchers, away_pitchers])

def convert_moneyline_into_prob(ml):
    ml = int(ml)
    if ml > 0:
        return 1/((ml/100) + 1)
    elif ml < 0:
        return 1/((100/abs(ml)) + 1)

class SolverError(Exception):
    pass

#Loading webscraping functions 
__path_to_prediction_script__ = '../prediction/'
sys.path.append(__path_to_prediction_script__)
import run_dk_predictions as dkp

slate_date = '2017-04-04'
# slate_date = str(datetime.now()).split()[0]
downloads = '../../Downloads/'

#Getting most recent DKSalaries.csv
path_to_dk_salaries = most_recent_dk_salaries(downloads)
# path_to_dk_salaries = '../../Downloads/DKSalaries (3).csv'
slate = pd.read_csv(path_to_dk_salaries)

#Gathering Starting Lineups and Vegas Lines 
print('Generating Optimal MLB Lineup for {}'.format(slate_date))
# lineups = dkp.get_and_parse_daily_lineups(slate_date)
lineups = scrape_roto_grinders(slate_date)
# pprint(lineups)
complete_lineups = {k: v for k, v in lineups.items() if v['home_lineup'] != []}
print('\n{} of {} lineups confirmed...'.format(len(complete_lineups), len(lineups)))

vegas_lines = dkp.scrape_mlb_vegas_lines(slate_date)
# print('\nVegas lines')
# print(vegas_lines)

#Subsetting slate to confirmed starters 
starters = get_starters(lineups)
starting = slate[slate.Name.isin(starters)]
print('\n{} starters confirmed'.format(len(starters)))

#Splitting up starters into pitchers and hitters
pitchers = starting[starting.Position == 'SP']
hitters = starting[starting.Position != 'SP']

# Getting Batting Order 
hitters['batting_order'] = 0
for i, row in hitters.iterrows():
    for game, info in complete_lineups.items():
        if row.Name in info['away_lineup']:
            hitters.loc[i, 'batting_order'] = info['away_lineup'].index(str(row.Name)) + 1
        elif row.Name in info['home_lineup']:
            hitters.loc[i, 'batting_order'] = info['home_lineup'].index(row.Name) + 1

#Matching up hitters with opposing pitchers
pitcher_dict = get_pitcher_dict(lineups)

#Taking care of multiple team abbreviations 
# pitcher_dict['MIA@WAS'] = pitcher_dict['MIA@WSH']
# pitcher_dict['DET@CWS'] = pitcher_dict['DET@CHW']
# pitcher_dict['KC@MIN '] = pitcher_dict['KCR@MIN']
pitcher_dict['SD@LAD '] = pitcher_dict['SDP@LAD']
pitcher_dict['SF@ARI '] = pitcher_dict['SFG@ARI']
pitcher_dict['NYY@TB '] = pitcher_dict['NYY@TBR']


hitters['Pitcher'] = hitters.apply(lambda row: 
    pitcher_dict[row.GameInfo.upper()[:7]][1] if row.teamAbbrev.upper() == row.GameInfo.split('@')[0].upper()
    else pitcher_dict[row.GameInfo.upper()[:7]][0], axis=1)

hitters = hitters.merge(pitchers, left_on='Pitcher', right_on='Name', suffixes=['_batter', '_pitcher'])

hitters.drop([ 
        'AvgPointsPerGame_batter',
        'Position_pitcher',
        'Name_pitcher',
        'GameInfo_pitcher',
        'AvgPointsPerGame_pitcher'
], axis=1, inplace=1)

hitters.columns = [
    'position',
    'batter',
    'batter_salary',
    'game_info',
    'batter_team',
    'batting_order',
    'pitcher',
    'pitcher_salary',
    'pitcher_team'
]

#Scaling salaries to be in thousands 
hitters.batter_salary = hitters.batter_salary.apply(lambda x: float(x)/1e3)
hitters.pitcher_salary = hitters.pitcher_salary.apply(lambda x: float(x)/1e3)

hitters['batter_home'] = hitters.apply(lambda row:
    1 if row.batter_team.upper() == row.game_info.split()[0][-3:].upper() else 0, axis=1)

hitters.drop('game_info', axis=1, inplace=1)

#Joining the hitters dataframe with vegas lines
vegas_lines['home_pitcher_throws'] = vegas_lines.home_pitcher.apply(
    lambda x: x[-3:].replace('(', '').replace(')', ''))
vegas_lines['away_pitcher_throws'] = vegas_lines.away_pitcher.apply(
    lambda x: x[-3:].replace('(', '').replace(')', ''))

vegas_lines['home_pitcher'] = vegas_lines.home_pitcher.apply(lambda x: x[:-3])
vegas_lines['away_pitcher'] = vegas_lines.away_pitcher.apply(lambda x: x[:-3])

#Fixing vegas lines (mispelled pitcher names, missing money lines, etc.)
vegas_lines.loc[5, 'home_pitcher'] = 'Perez'
vegas_lines.loc[8, 'home_ml'] = '+115'
vegas_lines.loc[8, 'away_ml'] = '-125'
print('Vegas Lines')
print(vegas_lines)

stacked_vegas_lines = stack_vegas_lines_by_pitcher(vegas_lines)

hitters['pitcher_last_name'] = hitters.pitcher.apply(lambda x: x.split()[1])
hitters_ = hitters.merge(stacked_vegas_lines, left_on='pitcher_last_name', right_on='pitcher', how='left')

hitters_['pitcher_throws'] = hitters_.apply(lambda row: 
    row.away_pitcher_throws if row.batter_home else row.home_pitcher_throws, axis=1)

hitters_.drop([
        'pitcher_last_name',
        'month_day',
        'pitcher_y',
        'home_pitcher_throws',
        'away_pitcher_throws'
], axis=1, inplace=1)

hitters_.rename(index=str, columns={'pitcher_x': 'pitcher'}, inplace=1)

# print(hitters_[hitters_.home_ml.isnull()])

hitters_ = hitters_[-hitters_.home_ml.isnull()]

#Converting money lines into probabilities 
hitters_['home_team_win_probability'] = hitters_.home_ml.apply(convert_moneyline_into_prob)
hitters_['away_team_win_probability'] = hitters_.away_ml.apply(convert_moneyline_into_prob)
hitters_.drop(['home_ml', 'away_ml'], axis=1, inplace=1)

#Getting BvP stats and splits
master = pd.read_csv('../mlb_data/master_v3.csv')
combos = [['R', 'R'], ['R', 'L'], ['L', 'R'], ['L', 'L']]
righty_bats = pd.read_csv('../mlb_data/splits/espn_splits/batting/RVR_batting_stats.csv')

hitters_['Batting Preference'] = hitters_.batter.apply(lambda x:
    'R' if x in righty_bats.PLAYER.values else 'L')
hitters_.rename(index=str, columns={'pitcher_throws': 'Pitching Preference'}, inplace=1)

#Loading MLB batting splits 
mlb_righty_splits = pd.read_csv('../mlb_data/splits/mlb.com/batters_against_RHP_mlb.csv')
mlb_lefty_splits = pd.read_csv('../mlb_data/splits/mlb.com/batters_against_LHP_mlb.csv')

mlb_righty_splits['Pitching Preference'] = 'R'
mlb_lefty_splits['Pitching Preference'] = 'L'
mlb_splits = pd.concat([mlb_righty_splits, mlb_lefty_splits])

#Joining Batters with splits by last name
mlb_splits['last_name'] = mlb_splits.Player.apply(lambda x: x.split(',')[0])
mlb_splits.rename(index=str, columns={'Team': 'batter_team'}, inplace=1)


hitters_['last_name'] = hitters_.batter.apply(lambda x: x.split()[1])
hitters_.batter_team = hitters_.batter_team.apply(lambda x: x.upper())
hitters2 = hitters_.merge(mlb_splits, on=['last_name', 'batter_team', 'Pitching Preference'], how='left')
print('Initial join on MLB splits:')
print("\t\tDimensions of hitters dataframe with NAs")
print('\t\t', hitters2.shape)
print("\t\tDimensions of hitters dataframe without NAs")
print('\t\t', hitters2.dropna().shape)

if hitters2.shape != hitters2.dropna().shape:
    print('\n\tChecking for traded players...')
    traded_players = hitters2[hitters2.AB.isnull()].batter.values
    if len(traded_players) > 0:
        print('\t\t{} traded players found'.format(len(traded_players)))
        print('\t\t', traded_players)
        traded = hitters_[hitters_.batter.isin(traded_players)].merge(
            mlb_splits, on=['last_name', 'Pitching Preference'], how='left')
        traded.drop('batter_team_y', axis=1, inplace=1)
        traded.rename(index=str, columns={'batter_team_x': 'batter_team'}, inplace=1)
        hitters2 = pd.concat([hitters2, traded])
    print('After including {} traded players'.format(len(traded_players)))
    print("\t\tDimensions of hitters dataframe with NAs")
    print('\t\t', hitters2.shape)
    print("\t\tDimensions of hitters dataframe without NAs")
    print('\t\t', hitters2.dropna().shape)

hitters2 = hitters2.dropna()

#Getting pitching splits 
pitch_split_dfs = pd.DataFrame()
for combo in combos:
    pitch_split_df = pd.read_csv(
        '../mlb_data/splits/espn_splits/pitching/{}V{}_pitching_stats.csv'.format(combo[0], combo[1]))
    pitch_split_df['Pitching Preference'] = [combo[0]]*pitch_split_df.shape[0]
    pitch_split_df['Batting Preference'] = [combo[1]]*pitch_split_df.shape[0]
    pitch_split_dfs = pd.concat([pitch_split_dfs, pitch_split_df])

hitters3 = hitters2.merge(pitch_split_dfs, 
                         left_on=['pitcher', 'Pitching Preference', 'Batting Preference'],
                         right_on=['PLAYER', 'Pitching Preference', 'Batting Preference'], how='left')

#Calculations
X = hitters3.copy()

X['batter_dk_salary'] = X.batter_salary
X['pitcher_dk_salary'] = X.pitcher_salary
X['batter_splits_walk_rate'] = X.BB_x/X.AB
X['batter_splits_batting_average'] = X.AVG
X['batter_splits_slugging_percentage'] = X.SLG
X['batter_splits_on_base_percentage'] = X.OBP
X['batter_splits_on_base_plus_slugging'] = X.OPS
X['batter_splits_double_rate'] = X['2B']/X.AB
X['batter_splits_triple_rate'] = X['3B']/X.AB
X['batter_splits_home_run_rate'] = X.HR/X.AB
X['batter_splits_RBI_rate'] = X.RBI/X.AB
X['batter_splits_stolen_base_rate'] = X.SB/X.AB
X['batter_splits_strikeout_rate'] = X.SO_x/X.AB
X['pitcher_splits_walks_plus_hits_per_inning_pitched'] = X.WHIP
X['pitcher_splits_earned_run_average'] = X.ERA
X['pitcher_splits_walk_rate'] = X.BB_y/X.IP
X['pitcher_splits_strikeout_rate'] = X.SO_y/X.IP
X['pitcher_win_probability'] = X.apply(lambda row: 
                                       row.away_team_win_probability if row.batter_home 
                                       else row.home_team_win_probability, axis=1)
X['batter_win_probability'] = X.apply(lambda row: 
                                       row.home_team_win_probability if row.batter_home 
                                       else row.away_team_win_probability, axis=1)
X['batter_batting_order'] = X.batting_order

X.drop([
    'Player',
    'Player ID',
    'Position',
    'AB',
    'R_x',
    'H_x',
    '2B',
    '3B',
    'HR',
    'RBI',
    'BB_x',
    'SO_x',
    'SB',
    'CS',
    'AVG',
    'OBP',
    'SLG',
    'OPS',
    'IBB',
    'HBP',
    'SAC',
    'SF',
    'TB',
    'XBH',
    'GIDP',
    'GO',
    'AO',
    'GO_AO',
    'NP',
    'TPA',
    'PLAYER',
    'TEAM',
    'GP',
    'GS',
    'IP',
    'H_y',
    'R_y',
    'ER',
    'BB_y',
    'SO_y',
    'W',
    'L',
    'SV',
    'HLD',
    'BLSV',
    'WHIP',
    'ERA',
    'Pitching Preference',
    'home_team_win_probability',
    'away_team_win_probability',
    'Batting Preference',
    'last_name',
    'home_team',
    'away_team',
    'batter_team',
    'pitcher_team',
], axis=1, inplace=1)

training_columns = ['batter_batting_order', 'pitcher_dk_salary', 'batter_home', 'over_under', 'batter_dk_salary',
       'batter_splits_walk_rate', 'batter_splits_batting_average',
       'batter_splits_slugging_percentage', 'batter_splits_on_base_percentage',
       'batter_splits_on_base_plus_slugging', 'batter_splits_double_rate',
       'batter_splits_triple_rate', 'batter_splits_home_run_rate',
       'batter_splits_RBI_rate', 'batter_splits_stolen_base_rate',
       'batter_splits_strikeout_rate',
       'pitcher_splits_walks_plus_hits_per_inning_pitched',
       'pitcher_splits_earned_run_average', 'pitcher_splits_walk_rate',
       'pitcher_splits_strikeout_rate', 'pitcher_win_probability',
       'batter_win_probability']


#Okay, we finally have all the data we need to make predictions

#Training hitter model on master dataset 
print('Training hitter model...\n')
batter_model = RandomForestRegressor(n_estimators=300, max_depth=15, max_features=10)
batter_model.fit(master[training_columns], master.batter_dk_points)
X = X.dropna()
batter_predictions = batter_model.predict(X[training_columns])
batter_prediction_df = pd.DataFrame({'Player': X.batter, 
                                     'Position': X.position, 
                                     'Salary': X.batter_dk_salary, 
                                     'Prediction': batter_predictions})

#Whew! That was a lot of work but we finally have projections for the batters on the slate

#Now all I have to do is aggregate the dataframe to make predictions for pitchers and we'll be off to the races. 

#Aggregating data for pitcher model
pitchers_ = X[['pitcher'] + training_columns].groupby(['pitcher'], 
                                    as_index=False).mean()
pitchers_.drop('batter_batting_order', axis=1, inplace=1)
pitcher_master = master.groupby(['pitcher', 'date', 'pitcher_team', 'batter_team'], 
                                    as_index=False).mean()

pitcher_train_columns = list(pitchers_.columns)
pitcher_train_columns.remove('pitcher')

#Training pitcher model on master dataset
print('Training pitcher model...\n')
pitcher_model = RandomForestRegressor(n_estimators=300, max_depth=10, max_features='sqrt')
pitcher_model.fit(pitcher_master[pitcher_train_columns], pitcher_master.pitcher_dk_points)
pitcher_predictions = pitcher_model.predict(pitchers_[pitcher_train_columns])

pitcher_pred_df = pd.DataFrame({'Player': pitchers_.pitcher, 
                                     'Position': ['SP']*pitchers_.shape[0], 
                                     'Salary': pitchers_.pitcher_dk_salary, 
                                     'Prediction': pitcher_predictions})

print('Pitcher predictions...')
print(pitcher_pred_df)

predictions = pd.concat([batter_prediction_df, pitcher_pred_df])
predictions = predictions.drop_duplicates(subset=['Player'])

print('\nPlayers unable to assign predictions to:\n')
print([p for p in starting.Name.values if p not in predictions.Player.values])


# print('\nTravis Shaw:')
# print(predictions[predictions.Player == 'Travis Shaw'])

#droping players
# predictions = predictions[predictions.Player != 'Clayton Kershaw']
# predictions = predictions[predictions.Player != 'Jose Ramirez']
# predictions = predictions[predictions.Player != 'Justin Verlander']
# predictions = predictions[predictions.Player != 'Justin Turner']

#Reformatting predictions df to work with protella_opt
predictions.reset_index(inplace=1)
multi_pos = predictions.loc[predictions.Position.str.contains('/')]
multi_pos1 = multi_pos.copy()
multi_pos2 = multi_pos.copy()
multi_pos1.Position = multi_pos.Position.apply(lambda x: x[:x.find('/')])
multi_pos2.Position = multi_pos.Position.apply(lambda x: x[x.find('/') + 1:])
split_predictions = predictions.loc[~predictions.Position.str.contains('/'), :].append(multi_pos1).append(multi_pos2)
split_predictions.loc[split_predictions.Position == 'SP', 'Position'] = 'P'
split_predictions.Salary = split_predictions.Salary.apply(lambda x : int(1000 * x))
split_predictions.loc[:, ['Player', 'Position', 'Prediction', 'Salary']].to_csv(
    '/tmp/dk_mlb_predictions.csv', header=None, index=None)


#Optimization
print('Optimizing predictions...\n')

#FUCK ORTOOLS! run protella_opt!!!

cmd = 'javac LineupOptimizerDK_MLB.java && java LineupOptimizerDK_MLB'
result = check_output([cmd], shell=True)
lineups = ast.literal_eval(result.decode())
optimal_lineup = lineups[-1]
optimal_players = [p.split('(')[0] for p in optimal_lineup]
optimal_player_positions = {p.split('(')[0]: p.split('(')[1].split(')')[0] for p in optimal_lineup}

optimal_df = split_predictions[split_predictions.Player.isin(optimal_players)]

# optimal_df['optimal_position'] = optimal_df.Player.apply(lambda x: optimal_player_positions[x])
# optimal_df['keep'] = optimal_df.apply(lambda row: 1 if row.Position == row.optimal_position else 0, axis=1)

# optimal_df = optimal_df[optimal_df.Position == optimal_df.optimal_position]

# optimal_df.drop('optimal_position', axis=1, inplace=1)

optimal_df.drop_duplicates(subset=['Player'], inplace=1)

#Hacking ortools import due to PYTHONPATH issues, don't ask...
# __path_to_ortools__ = '../../../../lib/python3.5/site-packages'
# sys.path.append(__path_to_ortools__)
# from ortools.linear_solver import pywraplp

# #TODO cycle through optimization with all combinations of pitchers 

# #Set optimization constraints
# SALARY_CAP = 50
# POSITION_LIMITS = [
#     ["SP", 2, 2],
#     ["C", 1, 1],
#     ["1B", 1, 1],
#     ["2B", 1, 1],
#     ["3B", 1, 1],
#     ["SS", 1, 1],
#     ["OF", 3, 3]
# ]
# ROSTER_SIZE = 10

# #Init solver and define variables
# solver = pywraplp.Solver('FD', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

# variables = [solver.IntVar(0, 1, row.Player) for _, row in predictions.iterrows()]


# objective = solver.Objective()
# objective.SetMaximization()

# [objective.SetCoefficient(variables[i], row.Salary) for i, row in predictions.iterrows()]

# salary_cap = solver.Constraint(0, SALARY_CAP)
# [salary_cap.SetCoefficient(variables[i], row.Salary) for i, row in predictions.iterrows()]

# for position, min_limit, max_limit in POSITION_LIMITS:
#     position_cap = solver.Constraint(min_limit, max_limit) 
#     for i, row in predictions.iterrows():
#         if position in row.Position:
#             position_cap.SetCoefficient(variables[i], 1)  

# size_cap = solver.Constraint(ROSTER_SIZE, ROSTER_SIZE)

# [size_cap.SetCoefficient(variable, 1) for variable in variables]

# solution = solver.Solve()

# roster = list()

# if solution == solver.OPTIMAL:
#     for i, row in predictions.iterrows():
#         if variables[i].solution_value() == 1:
#             roster.append({
#                 'Player': row.Player,
#                 'Position': row.Position,
#                 'DK Salary': '$' + str(int(row.Salary*1000)),
#                 'Projection': round(row.Prediction, 2)
#                 })
# else:
#     raise SolverError('Google-ortools was unable to find an optimal lineup. Check constraints and try again.')

print('Optimization complete! Optimal MLB Lineup for {}:\n'.format(slate_date))
print(tabulate.tabulate(optimal_df, tablefmt="psql", headers="keys"))
print('\nTotal Projected Points:', optimal_df.Prediction.sum())
print('Total Salary Spent:', optimal_df.Salary.sum())

