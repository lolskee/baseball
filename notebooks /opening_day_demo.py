
# coding: utf-8

# # My Quest To Conquer The World Of Baseball DFS - Part 3
# 
# ## Opening Day MLB 2017 DraftKings Optimization Demo
# 
# #### Tate Campbell - April 3, 2017

# Welcome back! We finally made it to opening day of the 2017 MLB season! I didn't have time to write up an article this week so I thought I'd just walk you through my first attempt at optimizing a lineup. The first and most daunting task will be transforming the DraftKings salary csv into the form of my training data. I still haven't finished the script that will ultimately do this for me, so there's going to be lots of parsing and joins this time through.
# 
# My first task is to gather all of the features used in my models. 
# 
# 

# ## Let's get started

# ### Loading Libraries 

# In[2]:


# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')
# import matplotlib.pyplot as plt
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
warnings.filterwarnings("ignore")


# In[3]:

__path_to_prediction_script__ = '../prediction/'
sys.path.append(__path_to_prediction_script__)
import run_dk_predictions as dkp


# ### Loading the DKSalaries.csv for Monday, April 3 MLB slate into a pandas dataframe 

# In[4]:

slate = pd.read_csv('../../Downloads/DKSalaries (3).csv')


# In[5]:

slate.head()


# ### Gathering Starting Lineups and Vegas Lines 

# In[6]:

slate_date = '2017-04-03'


# In[7]:

lineups = dkp.get_and_parse_daily_lineups(slate_date)


# In[8]:

lineups


# In[9]:

def get_starters(lineups):
    starters = list()
    for game, info in lineups.items():
        starters.append(info['away_SP'])
        starters.append(info['home_SP'])
        if len(info['away_lineup']) == 9:
            starters += info['away_lineup']
        if len(info['home_lineup']) == 9:
            starters += info['home_lineup']
    return starters


# In[10]:

def get_pitcher_dict(lineups):
    return {game: [info['away_SP'], info['home_SP']] for game, info in lineups.items()}


# In[11]:

complete_lineups = {k: v for k, v in lineups.items() if v['home_lineup'] != []}


# In[12]:

complete_lineups


# In[13]:

vegas_lines = dkp.scrape_mlb_vegas_lines('2017-04-03')


# In[14]:

vegas_lines


# ### Subsetting slate to confirmed starters 

# In[15]:

starting = slate.copy()


# In[16]:

starters = get_starters(lineups)


# In[17]:

starters


# In[18]:

starting = slate[slate.Name.isin(starters)]


# In[19]:

starting


# ### Splitting up starters into pitchers and hitters

# In[20]:

pitchers = starting[starting.Position == 'SP']
hitters = starting[starting.Position != 'SP']


# ### Getting Batting Order 

# In[21]:

hitters['batting_order'] = 0


# In[22]:

for i, row in hitters.iterrows():
    for game, info in complete_lineups.items():
#         print(info['away_lineup'])
#         print(row.Name)
#         print(row.Name in info['away_lineup'])
        if row.Name in info['away_lineup']:
            hitters.loc[i, 'batting_order'] = info['away_lineup'].index(str(row.Name)) + 1
        elif row.Name in info['home_lineup']:
            hitters.loc[i, 'batting_order'] = info['home_lineup'].index(row.Name) + 1


# In[23]:

hitters


# In[24]:

pitcher_dict = get_pitcher_dict(lineups)


# In[25]:

pitcher_dict['MIA@WAS'] = pitcher_dict['MIA@WSH']


# ### Matching up hitters with opposing pitchers

# In[26]:

hitters['Pitcher'] = hitters.apply(lambda row: 
    pitcher_dict[row.GameInfo.upper()[:7]][1] if row.teamAbbrev.upper() == row.GameInfo.upper()[:3]
    else pitcher_dict[row.GameInfo.upper()[:7]][0], axis=1)


# In[27]:

hitters


# In[28]:

hitters = hitters.merge(pitchers, left_on='Pitcher', right_on='Name', suffixes=['_batter', '_pitcher'])


# In[29]:

hitters.drop([ 
        'AvgPointsPerGame_batter',
        'Position_pitcher',
        'Name_pitcher',
        'GameInfo_pitcher',
        'AvgPointsPerGame_pitcher'
], axis=1, inplace=True)


# In[30]:

hitters


# In[31]:

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


# In[32]:

hitters.batter_salary = hitters.batter_salary.apply(lambda x: float(x)/1e3)
hitters.pitcher_salary = hitters.pitcher_salary.apply(lambda x: float(x)/1e3)


# In[33]:

hitters


# In[34]:

hitters['batter_home'] = hitters.apply(lambda row:
    1 if row.batter_team.upper() == row.game_info.split()[0][-3:].upper() else 0, axis=1)


# In[35]:

hitters.drop('game_info', axis=1, inplace=True)


# In[36]:

hitters


# ### Joining the hitters dataframe with vegas lines

# In[37]:

vegas_lines['home_pitcher_throws'] = vegas_lines.home_pitcher.apply(
    lambda x: x[-3:].replace('(', '').replace(')', ''))
vegas_lines['away_pitcher_throws'] = vegas_lines.away_pitcher.apply(
    lambda x: x[-3:].replace('(', '').replace(')', ''))


# In[38]:

vegas_lines['home_pitcher'] = vegas_lines.home_pitcher.apply(lambda x: x[:-3])


# In[39]:

vegas_lines['away_pitcher'] = vegas_lines.away_pitcher.apply(lambda x: x[:-3])


# In[40]:

def stack_vegas_lines_by_pitcher(vegas_lines):
    home_pitchers = vegas_lines.drop('away_pitcher', axis=1)
    away_pitchers = vegas_lines.drop('home_pitcher', axis=1)
    home_pitchers.rename(index=str, columns={"home_pitcher": "pitcher"}, inplace=True)
    away_pitchers.rename(index=str, columns={"away_pitcher": "pitcher"}, inplace=True)
    return pd.concat([home_pitchers, away_pitchers])


# In[41]:

stacked_vegas_lines = stack_vegas_lines_by_pitcher(vegas_lines)


# In[42]:

stacked_vegas_lines


# In[43]:

hitters['pitcher_last_name'] = hitters.pitcher.apply(lambda x: x.split()[1])


# In[44]:

hitters


# In[45]:

hitters_ = hitters.merge(stacked_vegas_lines, left_on='pitcher_last_name', right_on='pitcher', how='left')


# In[46]:

# hitters_ = hitters.merge(vegas_lines, left_on='pitcher_last_name', right_on='home_pitcher', how='outer')


# In[47]:

hitters_['pitcher_throws'] = hitters_.apply(lambda row: 
    row.away_pitcher_throws if row.batter_home else row.home_pitcher_throws, axis=1)


# In[48]:

hitters_.drop([
        'pitcher_last_name',
        'month_day',
        'pitcher_y',
        'home_pitcher_throws',
        'away_pitcher_throws'
], axis=1, inplace=True)


# In[49]:

hitters_.rename(index=str, columns={'pitcher_x': 'pitcher'}, inplace=True)


# In[50]:

hitters_


# ### Converting money lines into probabilities 

# In[51]:

def convert_moneyline_into_prob(ml):
    ml = int(ml)
    if ml > 0:
        return 1/((ml/100) + 1)
    elif ml < 0:
        return 1/((100/abs(ml)) + 1)


# In[52]:

hitters_['home_team_win_probability'] = hitters_.home_ml.apply(convert_moneyline_into_prob)


# In[53]:

hitters_['away_team_win_probability'] = hitters_.away_ml.apply(convert_moneyline_into_prob)


# In[54]:

hitters_.drop(['home_ml', 'away_ml'], axis=1, inplace=True)


# In[55]:

hitters_


# ### Getting BvP stats and splits

# Since I'm pressed for time I'm going to cheat here.
# 
# If I have the specific BvP matchup in my master dataset I'm going to grab the data from there.
# 
# In future should grab each players stats indepedently, e.g. get Votto's splits against righties and Hellickson's splits against lefties. 
# 

# In[56]:

master = pd.read_csv('../mlb_data/master_v3.csv')


# #### Counting BvP matchups in master

# In[57]:

matchups_accounted_for = 0
for _, row in hitters_.iterrows():
    sub = master[(master.batter == row.batter) & (master.pitcher == row.pitcher)]
    if sub.empty:
        pass
    else:
        matchups_accounted_for += 1

print("{}/{} BvP matchups accounted for in master".format(matchups_accounted_for, hitters_.shape[0]))


# On second thought, I'm not going to cheat. 
# 
# Getting data for only 21 of 32 batters isn't going to cut it. 

# ### Gettting batting stats and splits, the hard way

# In[58]:

combos = [['R', 'R'], ['R', 'L'], ['L', 'R'], ['L', 'L']]


# In[ ]:




# In[ ]:




# In[59]:

righty_bats = pd.read_csv('../mlb_data/splits/espn_splits/batting/RVR_batting_stats.csv')


# In[60]:

hitters_['Batting Preference'] = hitters_.batter.apply(lambda x:
    'R' if x in righty_bats.PLAYER.values else 'L')


# In[61]:

hitters_.rename(index=str, columns={'pitcher_throws': 'Pitching Preference'}, inplace=1)


# In[62]:

mlb_righty_splits = pd.read_csv('../mlb_data/splits/mlb.com/batters_against_RHP_mlb.csv')
mlb_lefty_splits = pd.read_csv('../mlb_data/splits/mlb.com/batters_against_LHP_mlb.csv')


# In[63]:

mlb_lefty_splits.sample(5)


# In[64]:

mlb_righty_splits['Pitching Preference'] = 'R'
mlb_lefty_splits['Pitching Preference'] = 'L'


# In[65]:

mlb_splits = pd.concat([mlb_righty_splits, mlb_lefty_splits])


# In[66]:

mlb_splits['last_name'] = mlb_splits.Player.apply(lambda x: x.split(',')[0])


# In[67]:

mlb_splits.rename(index=str, columns={'Team': 'batter_team'}, inplace=1)


# In[68]:

hitters_['last_name'] = hitters_.batter.apply(lambda x: x.split()[1])


# In[69]:

hitters_.batter_team = hitters_.batter_team.apply(lambda x: x.upper())


# In[70]:

hitters2 = hitters_.merge(mlb_splits, on=['last_name', 'batter_team', 'Pitching Preference'], how='left')


# In[71]:

print("Dimensions of hitters dataframe with NAs")
hitters2.shape


# In[72]:

print("Dimensions of hitters dataframe without NAs")
hitters2.dropna().shape


# Only 1 batter with no data, yay!

# In[73]:

hitters2.columns


# ### Getting pitching splits 

# In[74]:

pitch_split_dfs = pd.DataFrame()
for combo in combos:
    pitch_split_df = pd.read_csv(
        '../mlb_data/splits/espn_splits/pitching/{}V{}_pitching_stats.csv'.format(combo[0], combo[1]))
    pitch_split_df['Pitching Preference'] = [combo[0]]*pitch_split_df.shape[0]
    pitch_split_df['Batting Preference'] = [combo[1]]*pitch_split_df.shape[0]
    pitch_split_dfs = pd.concat([pitch_split_dfs, pitch_split_df])


# In[75]:

hitters2.head()


# In[76]:

pitch_split_dfs


# In[77]:

hitters3 = hitters2.merge(pitch_split_dfs, 
                         left_on=['pitcher', 'Pitching Preference', 'Batting Preference'],
                         right_on=['PLAYER', 'Pitching Preference', 'Batting Preference'], how='left')


# In[78]:

hitters3


# In[79]:

X = hitters3.copy()


# ### Calculations

# In[80]:

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


# In[82]:

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


# In[83]:

X.columns


# In[84]:

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


# # Okay, we finally have all the data we need to make predictions

# ### Training a model on master dataset 

# In[85]:

batter_model = RandomForestRegressor(n_estimators=300, max_depth=15, max_features=10)


# In[86]:

master.columns


# In[87]:

batter_model.fit(master[training_columns], master.batter_dk_points)


# In[88]:

X = X.dropna()


# In[89]:

batter_predictions = batter_model.predict(X[training_columns])


# In[90]:

batter_prediction_df = pd.DataFrame({'Player': X.batter, 
                                     'Position': X.position, 
                                     'Salary': X.batter_dk_salary, 
                                     'Prediction': batter_predictions})


# In[91]:

batter_prediction_df


# ## Whew! That was a lot of work but we finally have projections for the batters on the slate

# Now all I have to do is aggregate the dataframe to make predictions for pitchers and we'll be off to the races. 

# In[92]:

pitchers_ = X[['pitcher'] + training_columns].groupby(['pitcher'], 
                                    as_index=False).mean()


# In[93]:

pitchers_.drop('batter_batting_order', axis=1, inplace=1)


# In[94]:

pitcher_master = master.groupby(['pitcher', 'date', 'pitcher_team', 'batter_team'], 
                                    as_index=False).mean()


# In[95]:

pitcher_train_columns = list(pitchers_.columns)


# In[96]:

pitcher_train_columns.remove('pitcher')


# In[97]:

pitcher_train_columns


# ### Training Pitcher model on master 

# In[98]:

pitcher_model = RandomForestRegressor(n_estimators=300, max_depth=10, max_features='sqrt')


# In[99]:

pitcher_model.fit(pitcher_master[pitcher_train_columns], pitcher_master.pitcher_dk_points)


# In[100]:

pitcher_predictions = pitcher_model.predict(pitchers_[pitcher_train_columns])


# In[101]:

pitcher_pred_df = pd.DataFrame({'Player': pitchers_.pitcher, 
                                     'Position': ['SP']*pitchers_.shape[0], 
                                     'Salary': pitchers_.pitcher_dk_salary, 
                                     'Prediction': pitcher_predictions})


# In[102]:

pitcher_pred_df


# In[103]:

predictions = pd.concat([batter_prediction_df, pitcher_pred_df])


# # The predictions!

# In[104]:

predictions


# # Onto optimization 

# In[105]:

#Hacking ortools import due to PYTHONPATH issues, don't ask...
__path_to_ortools__ = '../../../../lib/python3.5/site-packages'
sys.path.append(__path_to_ortools__)
import ortools


# In[106]:

from ortools.linear_solver import pywraplp


# ### Optimizing 

# In[107]:

SALARY_CAP = 50
POSITION_LIMITS = [
    ["SP", 2, 2],
    ["C", 1, 1],
    ["1B", 1, 1],
    ["2B", 1, 1],
    ["3B", 1, 1],
    ["SS", 1, 1],
    ["OF", 3, 3]
]
ROSTER_SIZE = 10


# In[108]:

predictions = predictions.drop_duplicates(subset=['Player'])

solver = pywraplp.Solver('FD', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)


# In[1]:

variables = [solver.IntVar(0, 1, row.Player) for _, row in predictions.iterrows()]

objective = solver.Objective()
objective.SetMaximization()

[objective.SetCoefficient(variables[i], row.Salary) for i, row in predictions.iterrows()]

for position, min_limit, max_limit in POSITION_LIMITS:
    position_cap = solver.Constraint(min_limit, max_limit) 
    for i, row in predictions.iterrows():
        if position in row.Position:
            position_cap.SetCoefficient(variables[i], 1)  

size_cap = solver.Constraint(ROSTER_SIZE, ROSTER_SIZE)

[size_cap.SetCoefficient(variable, 1) for variable in variables]

solution = solver.Solve()

roster = list()

if solution == solver.OPTIMAL:
    for i, row in predictions.iterrows():
        if variables[i].solution_value() == 1:
            roster.append({
                'Player': row.Player,
                'Position': row.Position,
                'DK Salary': '$' + str(int(row.Salary*1000)),
                'Projection': round(row.Prediction, 2)
                })

roster_df = pd.DataFrame(roster)

print(tabulate.tabulate(roster, tablefmt="psql", headers="keys"))






# In[ ]:



