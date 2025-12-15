"""
NFLReadrPhaseI.py

Python translation of the R Markdown analysis in
`Final Project Phase 1 Markdown Report.Rmd`.

This script attempts to replicate data loading, integrity checks,
and EDA/plots using Python. It uses `nfl_data_py` where possible
and falls back to reading CSVs exported from `nflreadr`.

Usage:
    python NFLReadrPhaseI.py --team IND --season 2025

Outputs: prints summaries and shows plots.
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import nfl_data_py as nd
    HAVE_NFL_DATA_PY = True
except Exception:
    HAVE_NFL_DATA_PY = False

def load_schedules(season):
    if HAVE_NFL_DATA_PY:
        sched = nd.import_schedules([season])
        return sched
    # Fallback: expect schedules.csv in working dir
    return pd.read_csv('schedules.csv')

def load_pbp(season):
    if HAVE_NFL_DATA_PY:
        pbp = nd.import_pbp_data([season])
        return pbp
    return pd.read_csv('play_by_play.csv')

def load_rosters_weekly(season):
    if HAVE_NFL_DATA_PY:
        return nd.import_rosters([season])
    return pd.read_csv('rosters_weekly.csv')

def load_officials(season):
    # nfl_data_py does not have a dedicated officials loader; use schedules/plays
    if HAVE_NFL_DATA_PY:
        return nd.import_schedules([season])
    return pd.read_csv('officials.csv')

def load_players():
    if HAVE_NFL_DATA_PY:
        return nd.import_player_stats()
    return pd.read_csv('players.csv')

def main(team, season):
    pd.options.display.width = 200
    print(f"Team: {team} | Season: {season}")

    sched = load_schedules(season)
    sched.columns = sched.columns.str.lower()

    # Filter regular season and find first game for team
    colts_g1 = sched[(sched['gametype' if 'gametype' in sched.columns else 'game_type'] == 'REG')]
    # attempt columns
    if 'game_type' in sched.columns:
        colts_g1 = sched[(sched['game_type'] == 'REG') & (sched['season'] == season) & ((sched['home_team']==team) | (sched['away_team']==team))]
    else:
        # try lower-case alternatives
        cols = sched.columns
        possible_home = [c for c in cols if 'home' in c]
        possible_away = [c for c in cols if 'away' in c]
        if possible_home and possible_away:
            home_col = possible_home[0]
            away_col = possible_away[0]
            season_col = [c for c in cols if 'season' in c][0]
            colts_g1 = sched[(sched[season_col] == season) & ((sched[home_col]==team) | (sched[away_col]==team))]

    colts_g1 = colts_g1.sort_values(by=[c for c in ['gameday','week'] if c in colts_g1.columns])
    if len(colts_g1) == 0:
        print("No game found for team/season. Exiting.")
        return
    colts_g1 = colts_g1.iloc[0:1]

    # extract game attributes
    game_id = colts_g1['game_id'].iloc[0] if 'game_id' in colts_g1.columns else None
    week = colts_g1['week'].iloc[0] if 'week' in colts_g1.columns else None
    gameday = colts_g1['gameday'].iloc[0] if 'gameday' in colts_g1.columns else None
    home_tm = colts_g1['home_team'].iloc[0] if 'home_team' in colts_g1.columns else None
    away_tm = colts_g1['away_team'].iloc[0] if 'away_team' in colts_g1.columns else None
    opp = away_tm if home_tm == team else home_tm
    print(f"Found Week {week} ({gameday}): {away_tm} @ {home_tm} | game_id = {game_id}")

    pbp = load_pbp(season)
    pbp.columns = pbp.columns.str.lower()
    if game_id is not None and 'game_id' in pbp.columns:
        pbp_g1 = pbp[pbp['game_id'] == game_id].copy()
    else:
        pbp_g1 = pbp.copy()

    rw = load_rosters_weekly(season)
    rw.columns = rw.columns.str.lower()
    if week is not None and 'week' in rw.columns and 'team' in rw.columns:
        rw_g1 = rw[(rw['week']==week) & (rw['team'].isin([team, opp]))].copy()
    else:
        rw_g1 = rw.copy()

    officials = load_officials(season)
    officials.columns = officials.columns.str.lower()
    if game_id is not None and 'game_id' in officials.columns:
        officials_g1 = officials[officials['game_id']==game_id].copy()
    else:
        officials_g1 = officials.copy()

    players_master = load_players()
    players_master.columns = players_master.columns.str.lower()

    # Integrity checks summary
    print('\n--- STRUCTURE ---')
    print('PBP rows/cols:', pbp_g1.shape)
    print('Weekly roster rows/cols:', rw_g1.shape)
    print('Officials rows/cols:', officials_g1.shape)

    # duplicates in pbp
    if {'game_id','play_id'}.issubset(pbp_g1.columns):
        dup_plays = pbp_g1.groupby(['game_id','play_id']).size().reset_index(name='n')
        dup_plays = dup_plays[dup_plays['n']>1]
        if not dup_plays.empty:
            warnings.warn('Duplicate plays detected in (game_id, play_id).')
        else:
            print('Key check passed: (game_id, play_id) appear unique for this game.')

    # duplicates in roster
    roster_dup_cols = [c for c in ['team','gsis_id','week'] if c in rw_g1.columns]
    if roster_dup_cols:
        dup_roster = rw_g1.groupby(roster_dup_cols).size().reset_index(name='n')
        dup_roster = dup_roster[dup_roster['n']>1]
        if not dup_roster.empty:
            print('Note: roster has repeated (team, gsis_id, week) rows—can be normal.')

    # players present in master
    if 'gsis_id' in rw_g1.columns and 'gsis_id' in players_master.columns:
        rw_g1['in_master'] = rw_g1['gsis_id'].isin(players_master['gsis_id'])
        match_rate = rw_g1['in_master'].mean(skipna=True)
        print(f"Player master match rate (weekly roster -> master by gsis_id): {match_rate*100:.1f}%")
    else:
        print('gsis_id not present to check roster->player master referential integrity.')

    # Missingness
    def na_rate(x):
        return x.isna().mean()

    cols_to_check = {
        'na_epa':'epa', 'na_success':'success','na_posteam':'posteam','na_defteam':'defteam',
        'na_yardline_100':'yardline_100','na_down':'down','na_ydstogo':'ydstogo','na_play_type':'play_type'
    }
    na_summary = {k: na_rate(pbp_g1[v]) if v in pbp_g1.columns else None for k,v in cols_to_check.items()}
    print('\n--- MISSINGNESS (PBP) ---')
    print(na_summary)

    # yardline range
    if 'yardline_100' in pbp_g1.columns:
        bad_yardline = pbp_g1[pbp_g1['yardline_100'].notna() & ((pbp_g1['yardline_100']<0) | (pbp_g1['yardline_100']>100))]
        if not bad_yardline.empty:
            warnings.warn('Out-of-range yardline_100 values observed.')

    # down range
    if 'down' in pbp_g1.columns:
        bad_down = pbp_g1[pbp_g1['down'].notna() & (~pbp_g1['down'].isin([1,2,3,4]))]
        if not bad_down.empty:
            warnings.warn('Unexpected down values outside 1–4.')

    # ydstogo <= 0
    if 'ydstogo' in pbp_g1.columns:
        bad_ydstogo = pbp_g1[pbp_g1['ydstogo'].notna() & (pbp_g1['ydstogo'] <= 0)]
        if not bad_ydstogo.empty:
            print('Non-positive ydstogo rows exist (could be goal-to-go anomalies or data glitches).')

    if pbp_g1['game_id'].nunique() if 'game_id' in pbp_g1.columns else 1 != 1:
        warnings.warn('Multiple game_ids in pbp_g1; filter may be off.')

    # posteam/defteam correctness
    if {'posteam','defteam'}.issubset(pbp_g1.columns):
        valid_teams = [team, opp]
        weird_teams = pd.Series(pbp_g1['posteam'].dropna().unique())[~pd.Series(pbp_g1['posteam'].dropna().unique()).isin(valid_teams)]
        if len(weird_teams) > 0:
            warnings.warn('Found posteam not matching Colts or opponent: check weird_teams.')

    print('\n--- TEAM ABBREVS IN PBP ---')
    if 'posteam' in pbp_g1.columns:
        print(sorted(pbp_g1['posteam'].dropna().unique()))
    if 'defteam' in pbp_g1.columns:
        print(sorted(pbp_g1['defteam'].dropna().unique()))

    # EDA & Visualizations
    # points_scored column
    def points_scored_row(row):
        # best-effort mapping
        if 'touchdown' in row and row.get('touchdown') == 1:
            return 6
        if row.get('field_goal_result') == 'made':
            return 3
        if row.get('safety') == 1:
            return 2
        if row.get('extra_point_result') == 'good':
            return 1
        if row.get('two_point_attempt') == 1 and row.get('two_point_conv_result') == 'success':
            return 2
        return 0

    pbp_g1['points_scored'] = pbp_g1.apply(points_scored_row, axis=1)

    # score by quarter
    if 'qtr' in pbp_g1.columns:
        score_by_q = pbp_g1[pbp_g1['qtr'].isin([1,2,3,4])].groupby('qtr').apply(
            lambda df: pd.Series({
                'ind_points': df.loc[df['posteam']=='IND','points_scored'].sum() if 'posteam' in df.columns else 0,
                'opp_points': df.loc[df['posteam']==opp,'points_scored'].sum() if 'posteam' in df.columns else 0,
                'plays': len(df)
            })
        ).reset_index()
        print('\n--- SCORE BY QUARTER (derived from PBP points events) ---')
        print(score_by_q)

    # tempo
    if {'posteam','game_seconds_remaining'}.issubset(pbp_g1.columns):
        tempo = pbp_g1.dropna(subset=['posteam','game_seconds_remaining']).sort_values('game_seconds_remaining').groupby('posteam').apply(
            lambda df: pd.Series({'sec_per_play': df['game_seconds_remaining'].diff().median()})
        ).reset_index()
        tempo = tempo[tempo['sec_per_play'].notna()]
        print('\n--- TEMPO (median sec/play) ---')
        print(tempo)

    # offensive mix & efficiency
    if 'play_type' in pbp_g1.columns and 'posteam' in pbp_g1.columns:
        def play_family(pt):
            if pt == 'run':
                return 'Run'
            if pt == 'pass':
                return 'Pass'
            if pt in ('qb_kneel','qb_spike'):
                return 'Clock'
            if pt in ('no_play','timeout'):
                return 'Other'
            return 'Other'

        pbp_g1['play_family'] = pbp_g1['play_type'].map(play_family)
        off_mix = pbp_g1.groupby(['posteam','play_family']).agg(
            plays=('play_type','size'),
            epa_per_play=('epa','mean'),
            success_rate=('success','mean'),
            yards_per_play=('yards_gained','mean')
        ).reset_index()
        # share relative to full pbp_g1 rows
        off_mix['share'] = off_mix['plays'] / len(pbp_g1)
        print('\n--- OFFENSIVE MIX & EFFICIENCY ---')
        print(off_mix.head(50))

    # EPA distribution plot (pass & run)
    if {'posteam','play_type','epa'}.issubset(pbp_g1.columns):
        epa_dist = pbp_g1[pbp_g1['play_type'].isin(['run','pass']) & pbp_g1['posteam'].notna()]
        epa_dist['is_colts'] = np.where(epa_dist['posteam']==team, 'Colts', 'Opponent')
        plt.figure()
        sns.histplot(data=epa_dist, x='epa', hue='is_colts', bins=40, kde=False, alpha=0.6)
        means = epa_dist.groupby('is_colts')['epa'].mean().reset_index()
        for _, row in means.iterrows():
            plt.axvline(row['epa'], label=row['is_colts'], linestyle='--')
        plt.title('EPA distribution (pass & run plays)')
        plt.xlabel('EPA per play')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

    # success rate by yards to go and down
    if {'down','ydstogo','play_type','posteam','epa','success'}.issubset(pbp_g1.columns):
        situ = pbp_g1[pbp_g1['play_type'].isin(['pass','run']) & pbp_g1['down'].notna() & pbp_g1['ydstogo'].notna()].copy()
        def ytg_bucket(y):
            if y <= 2:
                return '1-2'
            if y <=5:
                return '3-5'
            if y <=10:
                return '6-10'
            return '11+'
        situ['ytg_bucket'] = situ['ydstogo'].apply(ytg_bucket)
        situ['down_lab'] = situ['down'].apply(lambda d: f'Down {int(d)}')
        situ_summary = situ.groupby(['posteam','down_lab','ytg_bucket']).agg(
            plays=('play_type','size'),
            success_rate=('success','mean'),
            epa_per_play=('epa','mean')
        ).reset_index()
        # plot for IND and opp
        plot_df = situ_summary[situ_summary['posteam'].isin([team, opp])]
        g = sns.FacetGrid(plot_df, col='down_lab', sharey=True)
        g.map_dataframe(sns.lineplot, x='ytg_bucket', y='success_rate', hue='posteam', marker='o')
        g.set_axis_labels('Yards to go (bucket)', 'Success rate')
        plt.show()

    # EPA/play by field zone
    if {'yardline_100','play_type','posteam','epa','success'}.issubset(pbp_g1.columns):
        def field_zone(y):
            if y >= 80:
                return 'Own 20-0'
            if y >= 50:
                return 'Own 49-21'
            if y >= 21:
                return 'Opp 49-21'
            return 'Red Zone (<=20)'
        fp = pbp_g1[pbp_g1['play_type'].isin(['run','pass']) & pbp_g1['yardline_100'].notna()].copy()
        fp['field_zone'] = fp['yardline_100'].apply(field_zone)
        fieldpos = fp.groupby(['posteam','field_zone']).agg(
            plays=('play_type','size'),
            epa_per_play=('epa','mean'),
            success_rate=('success','mean')
        ).reset_index()
        plot_df = fieldpos[fieldpos['posteam'].isin([team, opp])]
        plt.figure()
        sns.barplot(data=plot_df, x='epa_per_play', y='field_zone', hue='posteam')
        plt.title('EPA/play by field zone')
        plt.xlabel('EPA per play')
        plt.ylabel('')
        plt.show()

    # drive summaries
    if {'drive','posteam','yards_gained','epa','touchdown','field_goal_result','safety'}.issubset(pbp_g1.columns):
        def ended_score(df):
            return ((df['touchdown']==1) | (df['field_goal_result']=='made') | (df['safety']==1)).any()
        drive_sum = pbp_g1.dropna(subset=['drive','posteam']).groupby(['posteam','drive']).agg(
            plays=('play_type','size'),
            yards=('yards_gained','sum'),
            points=('points_scored','sum'),
            epa=('epa','sum'),
            success_rate=('success','mean')
        ).reset_index()
        drive_sum = drive_sum.sort_values(['posteam','points','yards'], ascending=[True,False,False])
        print('\n--- DRIVE SUMMARY ---')
        print(drive_sum.head())

    # receivers EPA vs targets
    if {'posteam','pass','receiver_player_name','epa'}.issubset(pbp_g1.columns):
        colts_targets = pbp_g1[(pbp_g1['posteam']==team) & (pbp_g1['pass']==1)].groupby('receiver_player_name').size().reset_index(name='targets')
        epa_by_receiver = pbp_g1[(pbp_g1['posteam']==team) & (pbp_g1['pass']==1) & pbp_g1['receiver_player_name'].notna()].groupby('receiver_player_name').agg(
            targets=('receiver_player_name','size'),
            epa_sum=('epa','sum'),
            epa_mean=('epa','mean')
        ).reset_index().sort_values('epa_sum', ascending=False)
        plt.figure()
        sns.scatterplot(data=epa_by_receiver, x='targets', y='epa_sum')
        for _, r in epa_by_receiver.iterrows():
            plt.text(r['targets'], r['epa_sum'], r['receiver_player_name'])
        plt.title("Colts receivers: total EPA vs targets")
        plt.show()

    # penalties and special teams
    if 'penalty' in pbp_g1.columns:
        penalties = pbp_g1[pbp_g1['penalty']==True] if pbp_g1['penalty'].dtype==bool else pbp_g1[pbp_g1['penalty'].notna()]
        if not penalties.empty:
            penalties['off_def'] = np.where(penalties['penalty_team']==penalties['posteam'],'Offense','Defense')
            pen_count = penalties.groupby(['penalty_team','off_def','penalty_type']).size().reset_index(name='n').sort_values('n', ascending=False)
            print('\n--- PENALTIES ---')
            print(pen_count.head(30))

    # EPA by quarter plot
    if {'qtr','play_type','posteam','epa'}.issubset(pbp_g1.columns):
        epa_by_q = pbp_g1[pbp_g1['qtr'].isin([1,2,3,4]) & pbp_g1['play_type'].isin(['pass','run'])].groupby(['posteam','qtr']).agg(epa_per_play=('epa','mean'), success_rate=('success','mean')).reset_index()
        plt.figure()
        sns.lineplot(data=epa_by_q, x='qtr', y='epa_per_play', hue='posteam', marker='o')
        plt.title('EPA/play by quarter')
        plt.show()

    # rolling pass completion rate (10 play window) for Colts
    if {'posteam','play_type'}.issubset(pbp_g1.columns):
        colts_seq = pbp_g1[pbp_g1['posteam']==team].copy()
        colts_seq = colts_seq[colts_seq['play_type'].isin(['pass','run'])]
        colts_seq = colts_seq.reset_index(drop=True)
        colts_seq['play_index'] = np.arange(1, len(colts_seq)+1)
        colts_seq['pass_flag'] = np.where(colts_seq['play_type']=='pass',1,0)
        if len(colts_seq) >= 10:
            colts_seq['pass_rate_10'] = colts_seq['pass_flag'].rolling(window=10, min_periods=1).mean()
            plt.figure()
            sns.lineplot(data=colts_seq, x='play_index', y='pass_rate_10')
            plt.title("Colts rolling pass rate (10-play window)")
            plt.ylabel('Pass rate')
            plt.show()

    # game summary
    if 'posteam' in pbp_g1.columns:
        colts_off = pbp_g1[pbp_g1['posteam']==team].agg(
            plays=('play_type','size'),
            pass_plays=('play_type', lambda s: (pbp_g1.loc[s.index,'play_type']=='pass').sum()),
            rush_plays=('play_type', lambda s: (pbp_g1.loc[s.index,'play_type']=='run').sum()),
            total_epa=('epa','sum'),
            mean_epa=('epa','mean'),
            success_rate=('success','mean')
        )
        plays_by_team_type = pbp_g1.groupby(['posteam','play_type']).size().reset_index(name='n').sort_values('n',ascending=False)
        print('\n--- SUMMARY ---')
        print({'game_header':{'season':season,'week':week,'gameday':gameday,'home':home_tm,'away':away_tm,'game_id':game_id,'opp_for_colts':opp},
               'pbp_rows':len(pbp_g1),'roster_rows':len(rw_g1),'officials_rows':len(officials_g1),'tempo_sec_per_play':None,'plays_by_team_type':plays_by_team_type.head(), 'colts_off_summary':colts_off.to_dict()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--team', type=str, default='IND')
    parser.add_argument('--season', type=int, default=2025)
    args = parser.parse_args()
    main(args.team, args.season)
