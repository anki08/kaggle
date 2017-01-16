# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:16:12 2016

@author: anki08
"""

import pandas as pd
import numpy as np

from scipy.stats.mstats import mode
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
severity_type = pd.read_csv('severity_type.csv')
resource_type = pd.read_csv('resource_type.csv')
log = pd.read_csv('log_feature.csv')
event = pd.read_csv('event_type.csv')
train['source'] = 'train'
test['source'] = 'test'
print(train.head())
print(train.info())
print(test.head())
print(test.info())
print(severity_type.head())
print(severity_type.info())
print(resource_type.head())
print(resource_type.info())
print(log.head())
print(log.info())
print(event.head())
print(event.info())
data = pd.concat([train, test], ignore_index=True)
print(data.head())
# find out no of unique values
# EVENT:
print(len(event['event_type'].unique()))
print(train['fault_severity'].value_counts())
event = event.merge(data, on='id')
print(event.head())
event_uniq = pd.DataFrame(event['event_type'].value_counts())
event_uniq['percent'] = event.pivot_table(
    values='source', index='event_type', aggfunc=lambda x: sum(x == 'train') / float(len(x)))
event_uniq['mode_severity'] = event.loc[event['source'] == 'train'].pivot_table(
    values='fault_severity', index='event_type', aggfunc=lambda x: mode(x).mode[0])
print(event_uniq.iloc[-15:])
top_unchange = 33
event_uniq['preprocess'] = event_uniq.index.values
event_uniq['preprocess'].iloc[top_unchange:] = event_uniq['mode_severity'].iloc[
    top_unchange:].apply(lambda x: 'Remove' if pd.isnull(x) else 'event_type others_%d' % int(x))
print(event_uniq.head())
print(event_uniq['preprocess'].value_counts())
print(event_uniq)
event = event.merge(event_uniq[['preprocess']],
                    left_on='event_type', right_index=True)
print(event.head())
event_merge = event.pivot_table(values='event_type', index='id',
                                columns='preprocess', aggfunc=lambda x: len(x), fill_value=0)
data = data.merge(event_merge, left_on='id', right_index='True')
print(data.head())
# LOG:
print(log['log_feature'].value_counts())
log = log.merge(data[['id', 'fault_severity', 'source']], on='id')
print(log.head())
log_unq = pd.DataFrame(log['log_feature'].value_counts())
print(log_unq.head())
log_unq['percent'] = log.pivot_table(
    values='source', index='log_feature', aggfunc=lambda x: sum(x == 'train') / float(len(x)))
log_unq['mode_severity'] = log.loc[log['source'] == 'train'].pivot_table(
    values='fault_severity', index='log_feature', aggfunc=lambda x: mode(x).mode[0])
log_unq['preprocess'] = log_unq.index.values
print(log_unq)
log_unq['preprocess'].loc[log_unq['percent'] == 1] = np.nan
top_unchange = 128
log_unq['preprocess'].iloc[top_unchange:] = log_unq['mode_severity'].iloc[
    top_unchange:].apply(lambda x: 'Remove' if pd.isnull(x) else 'feature others_%d' % int(x))
log = log.merge(log_unq[['preprocess']],
                left_on='log_feature', right_index=True)

log_merge = log.pivot_table(
    values='volume', index='id', columns='preprocess', aggfunc=np.sum, fill_value=0)
data = data.merge(log_merge, left_on='id', right_index=True)
print(data.head())
# RESOURCE_TYPE:
resource_type = resource_type.merge(
    data[['id', 'fault_severity', 'source']], on='id')
resource_type_unq = pd.DataFrame(resource_type['resource_type'].value_counts())
resource_type_unq['PercTrain'] = resource_type.pivot_table(
    values='source', index='resource_type', aggfunc=lambda x: sum(x == 'train') / float(len(x)))
resource_type_unq.head()
# Determine the mode of each:
resource_type_unq['Mode_Severity'] = resource_type.loc[resource_type['source'] == 'train'].pivot_table(
    values='fault_severity', index='resource_type', aggfunc=lambda x: mode(x).mode[0])
resource_type_merge = resource_type.pivot_table(
    values='source', index='id', columns='resource_type', aggfunc=lambda x: len(x), fill_value=0)
data = data.merge(resource_type_merge, left_on='id', right_index=True)
# SEVERITY_TYPE:

severity_type = severity_type.merge(
    data[['id', 'fault_severity', 'source']], on='id')
severity_type_unq = pd.DataFrame(severity_type['severity_type'].value_counts())
severity_type_unq['PercTrain'] = severity_type.pivot_table(
    values='source', index='severity_type', aggfunc=lambda x: sum(x == 'train') / float(len(x)))
severity_type_unq['Mode_Severity'] = severity_type.loc[severity_type['source'] == 'train'].pivot_table(
    values='fault_severity', index='severity_type', aggfunc=lambda x: mode(x).mode[0])

severity_type_merge = severity_type.pivot_table(
    values='source', index='id', columns='severity_type', aggfunc=lambda x: len(x), fill_value=0)
data = data.merge(severity_type_merge, left_on='id', right_index=True)
pred_event = [x for x in data.columns if 'event_type' in x]
pred_res = [x for x in data.columns if 'resource' in x]
pred_feat = [x for x in data.columns if 'feature' in x]
pred_sev = [x for x in data.columns if 'severity_type' in x]
location_count = data['location'].value_counts()
data['location_count'] = data['location'].apply(lambda x: location_count[x])
featvar = [x for x in data.columns if 'feature ' in x]
# log feature has volumes so add
data['feature_count'] = data[featvar].apply(np.sum, axis=1)
le = LabelEncoder()
data['location'] = le.fit_transform(data['location'])
[x for x in data.columns if 'Remove' in x]
data.drop(['Remove_x', 'Remove_y'], axis=1, inplace=True)
train_mod = data.loc[data['source'] == 'train']
test_mod = data.loc[data['source'] == 'test']
train_mod.drop('source', axis=1, inplace=True)
test_mod.drop(['source', 'fault_severity'], axis=1, inplace=True)
train_mod.to_csv('train_modified.csv', index=False)
test_mod.to_csv('test_modified.csv', index=False)
