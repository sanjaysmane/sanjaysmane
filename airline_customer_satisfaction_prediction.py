#!/usr/bin/env python
# coding: utf-8

# # Airline customer satisfaction problem: 
# ## 1. We are trying to figure what factors influences the satisfaction of the passengers.
# ## 2. Based on these factors we will predict if the passenger satisfy or not.
# 

# ## Import analysis libraries

# In[99]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':[8,6]}, font_scale = 1.2)


# ## Read dataset

# In[100]:


train = pd.read_csv('train.csv')
train


# In[101]:


test = pd.read_csv('test.csv')
test


# **Note:**
# > dataset is divided into **train set** and **test set**, we will merge them in one dataset in case of analysis.

# In[102]:


df = pd.concat([train, test])
del df['Unnamed: 0']
df


# ## Explore data

# In[103]:


df.info()


# **Notes:**
# > * Our dataset consists of 129880 rows and 24 columns.
# > * All feature datatypes are correct.

# *
# *Note:*
#     
# > All set, as a reminder we are trying to figure out what satisfy passengers.

# ## Data Analysis (EDA)

# In[104]:


import textwrap
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


# In[105]:


df['satisfaction'].value_counts()


# In[106]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
sns.countplot(data = df, x = 'satisfaction', palette='mako')
plt.title('Value count of satisfaction passengers')
ax.set_xlabel("Passenger satisfaction",fontsize=15, weight='semibold')
ax.set_ylabel("Count",fontsize=15, weight='semibold')
wrap_labels(ax, 10)


# In[107]:


var_cross = pd.crosstab(index = df.loc[:,"satisfaction"],columns = "count",normalize = False)


# In[108]:


plt.style.use("fivethirtyeight")
fig,ax = plt.subplots(figsize=(8,6))
var_cross.plot(kind = "bar",ax = ax,width = 0.3,legend = False)
plt.title('Value count of satisfactied passengers')
ax.set_xlabel("Passenger satisfaction",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.get_xticklabels()

plt.xticks(rotation = 0);


# **Notes:**
# > There is a noticeable difference here..! **neutral or dissatisfied passengers** is more than **satisfied passengers** in about **17024**.

# In[109]:


df['Gender'].value_counts()


# In[110]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
sns.countplot(data = df, x = 'Gender', palette='mako')
plt.title('Value count of gender feature')
ax.set_xlabel("Gender",fontsize=15, weight='semibold')
ax.set_ylabel("Count",fontsize=15, weight='semibold')
wrap_labels(ax, 10)


# In[111]:


var_cross1 = pd.crosstab(index = df.loc[:,"Gender"],columns = "count",normalize = False)


# **Notes:**
# > As we see the number of males and females almost the same there is a slightly difference.

# In[112]:


plt.style.use("fivethirtyeight")
fig,ax = plt.subplots(figsize=(8,6))
var_cross1.plot(kind = "bar",ax = ax,width = 0.3,legend = False)
plt.title('Value count of gender')
ax.set_xlabel("Gender",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.get_xticklabels()

plt.xticks(rotation = 0);


# In[113]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
sns.countplot(data = df, x = 'Gender', hue = 'satisfaction', palette='mako')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.title('Value count of gender feature')
ax.set_xlabel("Gender",fontsize=15, weight='semibold')
ax.set_ylabel("Count",fontsize=15, weight='semibold')
wrap_labels(ax, 10)


# In[114]:


var_cross2 = pd.crosstab(index = df.loc[:,"Gender"],columns = df.loc[:,"satisfaction"],normalize = False)


# In[115]:


fig,ax = plt.subplots(figsize=(8,6))
var_cross2.plot(kind = "bar",ax = ax,width = 0.3,legend =True)
plt.title('Genderwise count of satisfied / dissatisfied customers')
ax.set_xlabel("Gender",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.get_xticklabels()
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xticks(rotation = 0);


# **Note:**
# > As we saw before the count of males and females are the same. Here we can clearly see females are more unsatisfied than males.

# In[116]:


df['Customer Type'].value_counts()


# In[117]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
sns.countplot(data = df, x = 'Customer Type', palette='mako')
plt.title('Value count of customer type feature')
ax.set_xlabel("Customer Type",fontsize=15, weight='semibold')
ax.set_ylabel("Count",fontsize=15, weight='semibold')
wrap_labels(ax, 10)


# In[118]:


var_cross3 = pd.crosstab(index = df.loc[:,"Customer Type"],columns = "count",normalize = False)


# In[119]:


fig,ax = plt.subplots(figsize=(8,6))
var_cross3.plot(kind = "bar",ax = ax,width = 0.3,legend = False)
plt.title('Count of Customer Type')
ax.set_xlabel("Customer Type",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.get_xticklabels()

plt.xticks(rotation = 0);


# **Notes:**
# > Airline has more count of Loyal customers than Disloyal Customers. That is good news that loyal Customers fly with Airline.

# In[120]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
sns.countplot(data = df, x = 'Customer Type', hue = 'satisfaction', palette='mako')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.title('Value count of Customer Type feature with Satisfaction')
ax.set_xlabel("Customer Type",fontsize=15, weight='semibold')
ax.set_ylabel("Count",fontsize=15, weight='semibold')
wrap_labels(ax, 10)


# In[121]:


var_cross4 = pd.crosstab(index = df.loc[:,"Customer Type"],columns = df.loc[:,"satisfaction"],normalize = False)


# In[122]:


fig,ax = plt.subplots(figsize=(8,6))
var_cross4.plot(kind = "bar",ax = ax,width = 0.3,legend =True)
plt.title('Customer Type wise count of satisfied / dissatisfied customers')
ax.set_xlabel("Customer Type",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.get_xticklabels()
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xticks(rotation = 0);


# *Observations*
# >  As we can see Disloyal Customers are  more dissatisfied and that makes sense as they are not loyal to Airline.
# >  But in Loyal Customers also have more dissatisfaction and that's a problem.

# In[123]:


df['Age'].describe()


# **Notes:**
# > Average of ages here is **39 years old**.

# In[124]:


sns.displot(data = df['Age'], palette='mako', kde=True, height = 7)
plt.title('Distribution of Age')


# **Point to be Noted:**
#  * There are not too many childrens here.
#  * There is a noticeable increase in count when we come to age 25 years.
#  * From age 58 years we notice that the count come low.
#  * That means we are working in young people in age between (25 to 60) years. 

# In[125]:


df['Type of Travel'].unique()


# In[126]:


df['Type of Travel'].value_counts()


# In[127]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
sns.countplot(data = df, x = 'Type of Travel', palette='mako')
plt.title('Value count of Type of Travel feature')
ax.set_xlabel("Type of Travel",fontsize=15, weight='semibold')
ax.set_ylabel("Count",fontsize=15, weight='semibold')
wrap_labels(ax, 10)


# In[128]:


var_cross5 = pd.crosstab(index = df.loc[:,"Type of Travel"],columns = "count",normalize = False)


# In[129]:


fig,ax = plt.subplots(figsize=(8,6))
var_cross5.plot(kind = "bar",ax = ax,width = 0.3,legend =True)
plt.title(' count of  customers')
ax.set_xlabel("Travel Type",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.get_xticklabels()
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xticks(rotation = 0);


# *Notes:
# >  We have only 2 types of travel Personal Travel and Business Travel.
# >  Out Business Travel much more than Personal Travel.
# > That means we are working with Customers in Business Travel more.

# In[130]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
sns.countplot(data = df, x = 'Type of Travel', hue = 'satisfaction', palette='mako')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.title('Value count of Type of Travel feature with Satisfaction')
ax.set_xlabel("Type of Travel",fontsize=15, weight='semibold')
ax.set_ylabel("Count",fontsize=15, weight='semibold')
wrap_labels(ax, 10)


# In[131]:


var_cross6 = pd.crosstab(index = df.loc[:,"Type of Travel"],columns = df.loc[:,"satisfaction"],normalize = False)


# In[132]:


fig,ax = plt.subplots(figsize=(6,6))
var_cross6.plot(kind = "bar",ax = ax,width = 0.3,legend =True)
plt.title(' Travel Type wise count of satisfied/dissatisfied customers')
ax.set_xlabel("Travel Type",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.get_xticklabels()
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xticks(rotation = 0);


# **Notes:**
# > * We are not good at **Personal Travel** cuz the dissatisfied customers is much much much more than satisfied and that explains that **Business Travel more than Personal** we need to see what is the problems of **Personal Travel**.
# > * In **Business Travel** satisfied customers is much more than dissatisfied ones and that's good very good, but still No. of dissatisfied customers not low.

# In[133]:


df['Class'].unique()


# In[134]:


df['Class'].value_counts()


# In[135]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
sns.countplot(data = df, x = 'Class', palette='mako')
plt.title('Value count of Class feature')
ax.set_xlabel("Class",fontsize=15, weight='semibold')
ax.set_ylabel("Count",fontsize=15, weight='semibold')
wrap_labels(ax, 10)


# **Notes:**
# > * We have only 3 classes **Business - Eco - Eco Plus**.
# > * Almost no body uses **Eco Plus** class.

# In[136]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
sns.countplot(data = df, x = 'Class', hue = 'satisfaction', palette='mako')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.title('Value count of Class feature with Satisfaction')
ax.set_xlabel("Class",fontsize=15, weight='semibold')
ax.set_ylabel("Count",fontsize=15, weight='semibold')
wrap_labels(ax, 10)


# In[137]:


var_cross7 = pd.crosstab(index = df.loc[:,"Class"],columns = df.loc[:,"satisfaction"],normalize = False)


# In[138]:


fig,ax = plt.subplots(figsize=(6,6))
var_cross7.plot(kind = "bar",ax = ax,width = 0.3,legend =True)
plt.title(' Class wise count of satisfied/dissatisfied customers')
ax.set_xlabel("Travel Type",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.get_xticklabels()
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xticks(rotation = 0);


# **Notes:**
# > * In **Eco** class there is more passenger mad, let's see why by checking what **Type of Travel** in this class.

# In[139]:


fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([0,0,1,1])
sns.countplot(data = df, x = 'Class', hue = 'Type of Travel', palette='mako')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.title('Value count of Class feature with Satisfaction')
ax.set_xlabel("Class",fontsize=15, weight='semibold')
ax.set_ylabel("Count",fontsize=15, weight='semibold')
wrap_labels(ax, 10)


# In[140]:


var_cross8 = pd.crosstab(index = df.loc[:,"Class"],columns = df.loc[:,"Type of Travel"],normalize = False)


# In[141]:


fig,ax = plt.subplots(figsize=(6,6))
var_cross8.plot(kind = "bar",ax = ax,width = 0.3,legend =True)
plt.title(' Plot showing Class wise count ')
ax.set_xlabel("Class",fontsize=15)
ax.set_ylabel("Count",fontsize=15)
ax.get_xticklabels()
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xticks(rotation = 0);


# **Notes:**
# > * As we see in **Eco class** there is quite No. of **Business Travel** passenger take **Eco** and that not good for them cuz they want business qulity, so they dissatisfy.
# > * We need if the passenger in Business flight recommend for them **Business Class**.

# In[142]:


df['Flight Distance'].describe()


# **Notes:**
# > * Average of flight distances is 1190 km.
# > * And that means flights not take much time in air.

# **Notes:**
# > * We notice that we are working on small distance flights as the count of small kilometers flights much more than others.

# In[143]:


sns.displot(data = df['Flight Distance'], kde=True, height = 7)
plt.title('Distribution of Age');


# **Notes:**
# > Now we are gonna use rating and other measures to determine cases of satisfaction.

# In[144]:


def sub_bar_plots(cols, cat):
    plt.style.use("fivethirtyeight")
    fig, axis = plt.subplots(9, 2, figsize=(22, 64))
    fig.tight_layout(pad=3.0)
    for features, ax in zip(cols, axis.ravel()):
        ax = sns.barplot(data = df, x = 'satisfaction', y = features, hue = cat, ax = ax,)
        wrap_labels(ax, 10)


# In[145]:


cols = df.select_dtypes([np.number]).columns[2:]
cols


# In[146]:


sub_bar_plots(cols, 'Customer Type')


# *Oberservations:
# * Some obsevations on Loyal customers: 
#     * Satisfy with long distance flights.
#     * Not a much satisfied  with wifi service. 
#     * Arrival time and ease of booking have low rating. Problem lies here. Should be looked in details.
#     * Both types of customers  are satisfied with seat comfort and inflight entertainment so seems here no issue..
#     * Cleanliness most of them satify with Cleanliness but same time there is lot of dissatisfaction too. So we should look at this problem. 
#  
# *Observations on Disloyal customers:
#     * Doesn't matter if distance is big or not almost the same rating. 
#     * Satisfied with inflight wifi service, Ease of online booking, and Cleanliness.
#     
# *Both loyal and disloyal customers have no concern if flight Arrival_Time/Departure-Time delay time is up to 12.5 mins.

# In[147]:


sub_bar_plots(cols, 'Type of Travel')


# *Obervations:*
# >   Business travel:
#     * Customers who travels for business satisfy with long flight distance.
#     * Food and drink quality is good.So rating is good.
#     * Passengers traving for business are satisfied with seat comfort.
#     * They feel On-board service good as well.
# > * Personal travel:
#     * They seems to be happy with food qaulity.
#     * Rating of other parameters are same.
#     
# >  *As we can see the company is better for business traveler than personal travelers.

# In[148]:


sub_bar_plots(cols, 'Class')


# *Observations:*
#  * Airline excel at business class.
#  * Eco and Eco plus almost has same rating for all other services that Airline provides.

# ## Data Preprocessing

# In[149]:


df.isna().sum()


# **Notes:**
# > Arrival Delay in Minutes has missing values not that match big Nan values compare the size of dataset so we will dropna values.

# In[150]:


df = df.dropna()
df.isna().sum()


# **Notes:**
# > Clean, let's see correlation between features.

# In[151]:


fig = plt.figure(figsize=(25,10))
ax = fig.add_axes([0,0,1,1])
sns.heatmap(df.corr(), annot=True)
#plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.title('Correlation between features')
wrap_labels(ax, 1)


# In[152]:


del df['Departure Delay in Minutes']
df.info()


# *Notes:*
#  * There is a significat correlation between Departure Delay in Minutes and Arrival Delay in Minutes so we will drop one of 2 features to avoid multicolinearity problem.
#  * Encoding for categorical data, as our categorical columns not ordinal columns we will use getdummies function to encode them.

# In[153]:


df = pd.get_dummies(data = df, columns = ['Gender', 'Customer Type', 'Type of Travel', 
                                         'Class'], drop_first=True)
df


# 
# ** All good, we will split Data.
# ** We will check numeric columns's datatype is of correct datatype. if not will convert it.

# In[154]:


cols = ['id', 'Age', 'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Arrival Delay in Minutes',
       'Gender_Male', 'Customer Type_disloyal Customer',
       'Type of Travel_Personal Travel', 'Class_Eco', 'Class_Eco Plus']


# In[155]:


for i in cols:
    df[i] = pd.to_numeric(df[i], errors = 'coerce')
df


# In[156]:


df['satisfaction'].unique()


# In[157]:


dic = {'satisfied' : 1, 'neutral or dissatisfied': 0}
df['satisfaction'] = df['satisfaction'].map(dic)
df


# *Notes:*
# > * Now we have assigned to satisfy = 1, neutral or dissatisfied = 0.

# In[158]:


c = ['Gender_Male', 'Customer Type_disloyal Customer', 'Type of Travel_Personal Travel', 'Class_Eco', 'Class_Eco Plus']
for i in c:
    df[i] = df[i].astype(np.int)
    print(df[i].dtype)


# In[159]:


X = df.drop('satisfaction', axis = 1).iloc[:,1:]
y = df['satisfaction']
X


# In[170]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[161]:


get_ipython().system('pip install xgboost')


# ## Maching Learning Model

# In[171]:


from sklearn.linear_model import LogisticRegression


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, plot_roc_curve
import warnings

warnings.filterwarnings("ignore")


# In[172]:


logisreg_clf = LogisticRegression()

dt_clf = DecisionTreeClassifier(max_depth = 5)
rf_clf = RandomForestClassifier()
XGB_clf = XGBClassifier()


# In[173]:


clf_list = [logisreg_clf, dt_clf, rf_clf, XGB_clf]
clf_name_list = ['Logistic Regression', 'DecisionTree', 'Random Forest', 'XGBClassifier' ]
for clf in clf_list:
    clf.fit(X_train,y_train)


# In[183]:


train_acc_list = []
test_acc_list = []

for clf,name in zip(clf_list,clf_name_list):
    y_pred_test = clf.predict(X_test)
    print(f'Using model: {name}')
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred_test)}')
    print('                                             ')


# **Notes:**
# > Plot roc_curves to evaluate models.

# In[179]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
line = np.linspace(0,1)


sns.set(font_scale=1.0)
for clf, ax,name in zip(clf_list, axes.flatten(),clf_name_list):
    plot_roc_curve(clf, X_test, y_test, ax=ax, color = 'black')
    ax.plot(line, line, color='red', linestyle='dashed')
    ax.title.set_text(name)
fig.tight_layout(pad=1.0)
plt.show()


# In[184]:


from sklearn.metrics import classification_report
for clf,name in zip(clf_list,clf_name_list):
    print(classification_report(y_test,y_pred_test))
    
    

