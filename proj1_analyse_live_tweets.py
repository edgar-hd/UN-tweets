import os
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

def simplify_dates(datasetT):
    datasetT['Datetime'] = [datetime.datetime.strptime(re.sub('\+00:00','',i), date_format) for i in datasetT['Datetime']]
    return datasetT
    
def round_sentiments(datasetT):
    datasetT[['Negative','Neutral','Positive']] = np.rint(datasetT[['Negative','Neutral','Positive']])
    return datasetT

# This function imports the dataset and tidies up the sentiments a bit
def get_cleaned_df(file_read):
    dataset = pd.read_csv(file_read,lineterminator='\n')
    dataset = simplify_dates(dataset)
    dataset = round_sentiments(dataset)
    dataset = dataset.iloc[::-1].reset_index(drop=True)
    return dataset

# I found it easier to just have a single column with the sentiment associated to that tweet
def tidy_data(tweetdir,list_files):
    full_set = pd.concat([get_cleaned_df(tweet_dir + file) for file in list_files], ignore_index=True)
    full_set['Text'] = [re.sub('https.*','', line) for line in full_set['Text']]
    full_set = full_set.drop_duplicates(subset=['Text']).reset_index(drop=True)
    sentiment_column = np.full(len(full_set),'        ')
    sentiment_column[full_set['Negative'] == 1.0] = 'Negative'
    sentiment_column[full_set['Positive'] == 1.0] = 'Positive'
    sentiment_column[sentiment_column == '        '] = 'Neutral'
    full_set.insert(6, "Sentiment", sentiment_column, True)
    return(full_set)


tweet_dir = './data_tweets/'
date_format = '%Y-%m-%d %H:%M:%S'
list_files_raw = np.array(os.listdir(tweet_dir))
list_files = np.sort(list_files_raw[['un_mentions_processed' in file for file in list_files_raw]])
full_set = tidy_data(tweet_dir,list_files)

# Extract just the time and convert to hours
def get_day_counts(input_set):
    timeList = [timestamp.time() for timestamp in input_set]
    list_in_day = np.array([(t.hour * 60 + t.minute) * 60 + t.second for t in timeList])/3600
    return list_in_day

# Convert to day of the week and put on a range from 0 to 7 depending on day and time.
def get_week_counts(input_set):
    dayList = np.array([timestamp.weekday() for timestamp in input_set])
    list_in_week = dayList + get_day_counts(input_set)/24
    return list_in_week

# Get how many tweets happen in a week or day, to find patterns
list_all_day = get_day_counts(full_set['Datetime'])
list_all_week = get_week_counts(full_set['Datetime'])

# Group into a single dataframe for convenience
repeat_timescales = pd.concat([pd.DataFrame(list_all_day,columns=['Daily']),
                               pd.DataFrame(list_all_week,columns=['Weekly']),
                               full_set['Sentiment']],axis=1)


fig_dir = "figures/"

sns.set(rc = {'figure.figsize':(16,8)})
ax = sns.histplot(data=repeat_timescales, x='Daily', hue='Sentiment', stat="density", element="step")
ax.set_xlim([0,24]);
ax.set(xlabel='Time of the day', title='Distribution of tweets during the day');
plt.axvline(15, color='k', linestyle='dashed', linewidth=1);
plt.axvline(20, color='k', linestyle='dashed', linewidth=1);
plt.savefig(fig_dir+'fig1a_Day_distribution.pdf')

sns.set(rc = {'figure.figsize':(16,8)})
wPlot = sns.histplot(data=repeat_timescales, x='Weekly', hue='Sentiment', stat="density", element="step",)
wPlot.set_xticks(np.arange(0.5,7.5,1))
wPlot.set_xticklabels(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']);
wPlot.set_xlim([0,7]);
wPlot.set(title='Distribution of tweets during the week');
plt.savefig(fig_dir+'fig1b_Week_distribution.pdf')

# I use a sliding window to smooth out the data and analyse time intervals easier
def get_sliding_window_pos(start_date_string, window_size, step_size):
    num_hours = 24*(full_set['Datetime'].iloc[-1].date() - full_set['Datetime'].iloc[0].date() + datetime.timedelta(days = 1)).days
    start_date = datetime.datetime.strptime(start_date_string, date_format)
    window_index = np.array([[start_date + relativedelta(hours=i), 
                     start_date + relativedelta(hours=i*step_size) + relativedelta(hours=window_size)]
                    for i in range(num_hours - window_size)])
    return(window_index)


# This function collects the number of overall, as well as sentiment specific tweets on the sliding window
def get_window_number_exp_smth(window_index, alpha):
    window_index_result = []
    all_tweet_num = []
    for window_pos in window_index:
        window_index_result.append((window_pos[0] <= full_set['Datetime']) & (full_set['Datetime']  < window_pos[1]))
        window_tweets = full_set[(window_pos[0] <= full_set['Datetime']) & (full_set['Datetime']  < window_pos[1])]
        prev_weight = (all_tweet_num[-1] if len(all_tweet_num)>0 else [0,0,0,0,0])
        tot_tweets = alpha*len(window_tweets) + (1-alpha)*prev_weight[1]
        neg_tweets = alpha*len(window_tweets[window_tweets['Sentiment'] == 'Negative']) + (1 - alpha)*prev_weight[2]
        neu_tweets = alpha*len(window_tweets[window_tweets['Sentiment'] == 'Neutral']) + (1 - alpha)*prev_weight[3]
        pos_tweets = alpha*len(window_tweets[window_tweets['Sentiment'] == 'Positive']) + (1 - alpha)*prev_weight[4]
        all_tweet_num.append([window_pos[1],tot_tweets,neg_tweets,neu_tweets,pos_tweets])
    all_tweet_num = pd.DataFrame(all_tweet_num)
    all_tweet_num = all_tweet_num.set_axis(
        ['Datetime', 'All', 'Negative', 'Neutral', 'Positive'],
        axis=1,
        copy=False
    )
    return (all_tweet_num, window_index_result)


# proj1_script_c_get_sliding_window.py

start_date_string = '2021-01-01 00:00:00'
window_index = get_sliding_window_pos(start_date_string, window_size = 72, step_size = 1)
# window_tweet_num, window_index_result = get_window_number_sentiment(window_index)
window_tweet_num_smth, window_index_result = get_window_number_exp_smth(window_index,0.1)
print("Indexed sliding window tweets")

sns.set(rc = {'figure.figsize':(16,8)})
fig, ax = plt.subplots()
ax.stackplot(np.array(window_tweet_num_smth['Datetime']), np.array(window_tweet_num_smth[['Neutral','Negative','Positive']]).T,
            labels = ['Neutral','Negative','Positive'])
ax.set_xlim([datetime.datetime(2021, 1, 1), datetime.datetime(2022, 4, 30)]);
ax.set(ylabel='Number of tweets in 72h', title='Tweet volume over time', label=["Fibonacci ", "Evens", "Odds"]);
ax.legend(loc='upper left');
plt.savefig(fig_dir+'fig1c_All_tweets_dynamics.pdf')
print("Classified tweets by sentiment")

###################### New Section

import sys
import nltk
import string
from nltk import word_tokenize
from collections import Counter
from nltk.corpus import stopwords

print("Downloading stopwords")
nltk.download('stopwords')
nltk.download('punkt')

print("Set up stopwords")
stop = set(stopwords.words('english') + list(string.punctuation))
stop.update(['’','https','un','amp','``',"''","'s",'..','...',"n't",'--','”','–','//','“','like','also','put','ask','w/','unitednations'])
print("Defined stopwords")

def clean_sentence(sentence):
    cleaned = [i for i in word_tokenize(sentence.lower()) if i not in stop]
    return cleaned

words_List = [clean_sentence(text) for text in full_set['Text']]
print("Performed word cleaning")

def get_flat_window(window_index, condition):
    all_words = [words_List[i] for i in np.where(window_index & condition)[0]]
    flat_list_words = [item for sublist in all_words for item in sublist]
    dict_index = Counter(flat_list_words)
    return dict_index
    

def get_top10_progression(condition, top10_words, alpha):
    dict_index = get_flat_window(window_index_result[0], condition)
    top10_dyn = [np.array([dict_index.get(key,0) for key in top10_words])]
    
    for index in window_index_result[1:]:
        dict_index = get_flat_window(index, condition)
        temp_vals = np.array([dict_index.get(key,0) for key in top10_words])
        top10_dyn.append(alpha*temp_vals + (1-alpha)*top10_dyn[-1])
        
    print("For '"+str(condition)+"', obtained dynamics of top10 words")
    return(np.array(top10_dyn))


def get_top10_window(condition, min_times, alpha):
    top10_list = set()
    for window_pos in window_index_result:
        dict_index = get_flat_window(window_pos, condition)
        top_words = dict_index.most_common(10)
        top_words = [tup[0] for tup in top_words if tup[1] >= min_times]
        top10_list.update(top_words)

    top10_list = list(dict.fromkeys(top10_list))
    print("For '"+str(condition)+", obtained top10 words")
    top10_dyn = get_top10_progression(condition, top10_list, alpha)
    
    return(top10_list, top10_dyn)

min_times = 20
top10_pos, top10_dyn_pos = get_top10_window(full_set['Sentiment'] == 'Positive', min_times, 0.1)
top10_neg, top10_dyn_neg = get_top10_window(full_set['Sentiment'] == 'Negative', min_times, 0.1)
top10_neu, top10_dyn_neu = get_top10_window(full_set['Sentiment'] == 'Neutral', min_times, 0.1)
top10_tot, top10_dyn_tot = get_top10_window(1, min_times, 0.1)

print("All top10 word trajectories captured")

from sklearn import decomposition

print("Performing PCA analysis")

pca_tot = decomposition.PCA(n_components=0.95)
pca_tot.fit(top10_dyn_tot)
top10_dyn_tot_PCA = pca_tot.fit_transform(top10_dyn_tot)
var_comp_tot = pca_tot.explained_variance_ratio_[pca_tot.explained_variance_ratio_ >= 0.05]
print(var_comp_tot)

pca_topics_tot = [np.array(top10_tot)[np.flip(np.argsort(component))][:np.sum(np.flip(np.sort(component)) > 0.15)-1]
for component in pca_tot.components_]
top_topics_tot = [topics[0] for topics in pca_topics_tot][:len(var_comp_tot)]
top_topics_tot_PCA = ['PCA '+str(i+1)+': '+topics[0] for i, topics in enumerate(pca_topics_tot)][:len(var_comp_tot)]

plt.figure(0)
sns.set(rc = {'figure.figsize':(16,8)})
plt.plot(window_index[:,1],top10_dyn_tot_PCA[:,:len(var_comp_tot)]);
plt.gca().legend(top_topics_tot_PCA);
plt.xlim([datetime.datetime(2021, 1, 1), datetime.datetime(2022, 4, 30)]);
plt.ylabel("Component value"); plt.title("Principal Components over time");
plt.savefig(fig_dir+'fig2a_top_tot_PCA.pdf')

plt.figure(0)
sns.set(rc = {'figure.figsize':(16,8)})
top_words_tot_dyn = np.array([top10_dyn_tot[:,top10_tot.index(top_topics_tot[i])]
                              for i in range(len(top_topics_tot))]).T
plt.plot(window_index[:,0],top_words_tot_dyn)
plt.gca().legend(top_topics_tot)
plt.xlim([datetime.datetime(2021, 1, 1), datetime.datetime(2022, 4, 30)]);
plt.ylabel("# of tweets"); plt.title("Tweets over time");
plt.savefig(fig_dir+'fig2b_top_tot_from_PCA.pdf')


pca_pos = decomposition.PCA(n_components=0.95)
pca_pos.fit(top10_dyn_pos)
top10_dyn_pos_PCA = pca_pos.fit_transform(top10_dyn_pos)
var_comp_pos = pca_pos.explained_variance_ratio_[pca_pos.explained_variance_ratio_ >= 0.05]
print(var_comp_pos)

pca_topics_pos = [np.array(top10_pos)[np.flip(np.argsort(component))][:np.sum(np.flip(np.sort(component)) > 0.1)-1]
for component in pca_pos.components_]
top_topics_pos = [topics[0] for topics in pca_topics_pos][:len(var_comp_pos)]
top_topics_pos_PCA = ['PCA '+str(i+1)+': '+topics[0] for i, topics in enumerate(pca_topics_pos)][:len(var_comp_pos)]

plt.figure(0)
sns.set(rc = {'figure.figsize':(16,8)})
plt.plot(window_index[:,1],top10_dyn_pos_PCA[:,:len(var_comp_pos)]);
plt.gca().legend(top_topics_pos_PCA);
plt.xlim([datetime.datetime(2021, 1, 1), datetime.datetime(2022, 4, 30)]);
plt.ylabel("Component value"); plt.title('"Positive" Principal Components over time');
plt.savefig(fig_dir+'fig2c_top_pos_PCA.pdf')

plt.figure(0)
sns.set(rc = {'figure.figsize':(16,8)})
top_words_pos_dyn = np.array([top10_dyn_pos[:,top10_pos.index(top_topics_pos[i])]
                              for i in range(len(top_topics_pos))]).T
plt.plot(window_index[:,0],top_words_pos_dyn)
plt.gca().legend(top_topics_pos)
plt.xlim([datetime.datetime(2021, 1, 1), datetime.datetime(2022, 4, 30)]);
plt.ylabel("# of tweets"); plt.title("Positive tweets over time");
plt.savefig(fig_dir+'fig2d_top_pos_from_PCA.pdf')


###################### New Section

test = np.array([np.ceil(np.clip(feature / max(np.median(feature), 1) - 8, 0, 1)) for feature in top10_dyn_tot.T])
peak_rows = np.array([row for row in test if np.sum(row > 0)])
peak_row_names = np.array([top10_tot[i] for i,row in enumerate(test) if np.sum(row > 0)])
peak_row_index = np.array([i for i,row in enumerate(test) if np.sum(row > 0)])

plt.figure(0)
sns.set(rc = {'figure.figsize':(16,8)})
plt.plot(peak_rows.T);
plt.title("Number of salient topics: "+str(len(peak_rows)));
plt.savefig(fig_dir+'fig3a_onoff_topics.pdf')

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return linkage_matrix

def llf(id):
    if id <= 96:
        return str(peak_row_names[id])
    else:
        return '[%d %s]' % (id, 'beep')

peak_row_names[peak_row_names == '📺livestreaming'] = 'livestreaming'

import matplotlib
matplotlib.rc_file_defaults()
cos_dis_thrs = 0.7
clustering = AgglomerativeClustering(distance_threshold=cos_dis_thrs, n_clusters=None,
                                linkage='average', affinity='cosine', compute_distances=True)
clustering = clustering.fit(peak_rows)

plt.figure(0)
plt.figure(figsize=(16, 30))
lm = plot_dendrogram(clustering, color_threshold = cos_dis_thrs, labels = peak_row_names, orientation = 'right',
                    leaf_font_size=14)
plt.savefig(fig_dir+'fig3b_dendogram_trends.pdf')
plt.savefig(fig_dir+'fig3b_dendogram_trends.svg')
plt.savefig(fig_dir+'fig3b_dendogram_trends.png')


###################### New Section

plt.figure(0)
ax = sns.jointplot(x=np.sum(peak_rows,axis=1), y=np.sum(top10_dyn_tot[:,peak_row_index],axis=0)/24)
ax.set_axis_labels(xlabel="Hours actively tweeted", ylabel="Smoothed total tweet number");
plt.savefig(fig_dir+'fig4a_time-vs-volume.pdf')

def get_bursts(burst_list, masked_list, normal_list):
    start_burst,end_burst = [],[]
    for i, val in enumerate(masked_list[:-1]):
        if (masked_list[i] == 0) & (masked_list[i+1] > 0):
            start_burst.append(i+1)
        if (len(start_burst) > len(end_burst)) & (normal_list[i] <= 20) & (masked_list[i+1] == 0):
            end_burst.append(i+1)
    
    for i in range(len(end_burst)):
        burst_list.append(normal_list[start_burst[i]:end_burst[i]])
        
    return(burst_list)


burst_list = []
masked_dynamics = peak_rows * top10_dyn_tot[:,peak_row_index].T

for masked, normal in zip(masked_dynamics, top10_dyn_tot[:,peak_row_index].T):
    burst_list = get_bursts(burst_list, masked, normal)
    
burst_acum_list = [np.cumsum(burst) for burst in burst_list]


def get_ongoing_bursts(burst_list, masked_list, normal_list):
    start_burst,end_burst = [],[]
    for i, val in enumerate(masked_list[:-1]):
        if (masked_list[i] == 0) & (masked_list[i+1] > 0):
            start_burst.append(i+1)
        if (len(start_burst) > len(end_burst)) & (normal_list[i] <= 20) & (masked_list[i+1] == 0):
            end_burst.append(i+1)

    if len(start_burst) > len(end_burst):
        for start in start_burst:
            if (end_burst == []):
                burst_list.append(normal_list[start:])
                break
            elif (start > end_burst[-1]):
                burst_list.append(normal_list[start:])
                break
    
    return(burst_list)


ongoing_burst_list, ongoing_burst_index = [], []
temp = 0

for i, (masked, normal) in enumerate(zip(masked_dynamics, top10_dyn_tot[:,peak_row_index].T)):
    ongoing_burst_list = get_ongoing_bursts(ongoing_burst_list, masked, normal)
    if temp < len(ongoing_burst_list):
        ongoing_burst_index.append(i)
        temp = len(ongoing_burst_list)
    
ongoing_burst_acum_list = [np.cumsum(burst) for burst in ongoing_burst_list]


def get_assym_split(vec):
    space_vec = len(vec)*np.array([1-1/(50-1)*(50**i-1) for i in np.arange(0,1,0.1)])
    return np.flip(np.round(space_vec)-1).astype(np.int16)

min_split_size = 50
full_bursts = []
for (burst, acum_burst) in zip(burst_list, burst_acum_list):
    if len(burst) > min_split_size:
        num_sec = np.floor(np.log2(len(burst))).astype(np.int16)-2
        sec_size = np.floor(len(burst)/num_sec).astype(np.int16)
        for split in range(sec_size, len(burst)-num_sec, sec_size):
#             print(len(burst),num_sec,sec_size,split,len(burst[:-split]))
            training = np.concatenate([[len(burst[:-split])],
#                              [sec[-1] for sec in np.array_split(burst[:-split],10)],
#                              [sec[-1] for sec in np.array_split(acum_burst[:-split],10)],
                             [burst[sec] for sec in get_assym_split(burst[:-split])],
                             [acum_burst[sec] for sec in get_assym_split(acum_burst[:-split])],
                             [len(burst)]
                            ])
            full_bursts.append(training)

full_bursts = np.array(full_bursts)

from sklearn.model_selection import train_test_split
X, y = full_bursts[:,:-1], full_bursts[:,-1]

y = y/24
y = np.log2(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.clf()
plt.figure(0)
sns.set(rc = {'figure.figsize':(16,8)})
bx = sns.histplot(y);
bx.set_xlim([0.5,8.5])
bx.set_xticks(np.arange(1,9,1))
bx.set_xticklabels(np.exp2(np.arange(1,9,1)).astype(np.int16));
bx.set(title='Length of training peaks, total peaks: '+str(len(full_bursts)));
plt.savefig(fig_dir+'fig4b_peak_distribution.pdf')


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor

reg = ExtraTreesRegressor(n_estimators=1000, random_state=42).fit(X_train, y_train)
print(reg.score(X_test, y_test))

plt.clf()
plt.figure(0)
sns.set(rc = {'figure.figsize':(6,6)})
cx = sns.scatterplot(x=y_test,y=reg.predict(X_test));
# ax.set_xlim([0.5,8.5])
# ax.set_ylim([0.5,8.5])
cx.set(ylabel="Predicted Number of Days",xlabel="Real Number of Days",title="Regression Score: "+str(np.round(reg.score(X_test, y_test),2)));
cx.set_xticks(np.arange(1,9,1))
cx.set_xticklabels(np.exp2(np.arange(1,9,1)).astype(np.int16));
cx.set_yticks(np.arange(1,9,1))
cx.set_yticklabels(np.exp2(np.arange(1,9,1)).astype(np.int16));
plt.savefig(fig_dir+'fig4c_test_correlation.pdf')


ongoing_full_bursts = []
for (burst, acum_burst) in zip(ongoing_burst_list, ongoing_burst_acum_list):
    if len(burst) > 24:
        training = np.concatenate([[len(burst)],
#                         [sec[-1] for sec in np.array_split(burst,10)],
#                         [sec[-1] for sec in np.array_split(acum_burst,10)]
                        [burst[sec] for sec in get_assym_split(burst)],
                        [acum_burst[sec] for sec in get_assym_split(acum_burst)]
                        ])
        ongoing_full_bursts.append(training)

ongoing_full_bursts = np.array(ongoing_full_bursts)


minLen_index =[ongoing_burst_index[i] for i in range(len(ongoing_burst_list)) if len(ongoing_burst_list[i]) > 24]
ongoing_top_words = pd.DataFrame([peak_row_names[minLen_index],
    np.array([len(burst)/24 for burst in ongoing_burst_list if len(burst) > 24]),
    np.exp2(reg.predict(ongoing_full_bursts))]).transpose()
ongoing_top_words.columns = ["Word","Current Duration","Predicted Lifetime"]
melted_ongoing_top_words = ongoing_top_words.melt(id_vars='Word', var_name="Keys", value_name="Days")
melted_ongoing_top_words["Days"] = melted_ongoing_top_words["Days"].astype(np.float32)

plt.clf()
plt.figure(0)
sns.set(rc = {'figure.figsize':(16,8)})
dx = sns.barplot(data = melted_ongoing_top_words, x= "Word", y= "Days", hue= "Keys");
plt.xticks(rotation=45);
plt.savefig(fig_dir+'fig4d_current_topics_lifetime.pdf')
plt.savefig(fig_dir+'fig4d_current_topics_lifetime.png')