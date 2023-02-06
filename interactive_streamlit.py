#pip install streamlit-option-menu
#pip install streamlit-lottie
#pip install requests


import pandas as pd
import numpy as np
#from pandas_profiling import ProfileReport
import datetime
import streamlit as st
#import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from itertools import product
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.metrics import f1_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import BaggingClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.utils.validation import column_or_1d

import requests
import warnings
warnings.filterwarnings('ignore') # Do not print warning messages

data = pd.read_csv("data/bike-sharing_hourly.csv")

#create feature hr_str based on hr column for visualization purposes

data['hr_str'] = data.hr.astype(str) + ":00:00"

#create a column with datetime
data['Datetime'] = data['dteday'] +' ' + data['hr_str']
data['Datetime'] = pd.to_datetime(data['Datetime'] )
data['DayofWeek'] = data['weekday'].replace({1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 0: 'Sunday'})
#Streamlit page

def main():
    

#    sources = data.src_neigh_name.unique()
#    destinations = data.dst_neigh_name.unique()
#    SOURCE = st.sidebar.selectbox('Select Source', sources)
#    DESTINATION = st.sidebar.selectbox('Select Destination', destinations)

#    aux = data[(data.src_neigh_name == SOURCE) & (data.dst_neigh_name == DESTINATION)]
#    aux = aux.sort_values("date")

    
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    

#-------------------- temp vs atemp ----------------------

    with st.sidebar:
        
        selected = option_menu(menu_title = "Main Menu",
                               options = ["Home","Data", "Insights", "ML model"],
                               icons = ["house","table","file-bar-graph","robot"],
                               menu_icon = "cast",
                               default_index = 0,
                              )
        
    
    if selected == "Home":
        
        st.markdown("### **Bike-sharing Analysis - Washington DC**")
        
        lottie_bike0 = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_0PjKRV.json")
            
        left_column, right_column = st.columns(2)
            
        with left_column:
        
            st.write("""
            In this application, you will be able to understant the behavior of bike users in 2011-2012 in Washington DC. This app is splitted in three sections, the first one is to understand and explore the data about the bike users, the second section contains the business insights that we are proporsing to the administration of Washington DC iin order to increase the efficiency of the service. And finally, in the third section, you will be able to generate your own forecast over the total number of users that this state can have in an sepecific hour of the day.  
            """)
        
        with right_column:
        
            st_lottie(lottie_bike0, height=250, key = "bike0")
            
        st.markdown("**Let's start**")
                             
        
    
    if selected == "Data":
            
            st.title(f" Exploration of the {selected}")
            
            st.markdown("### Overview of the data")
            
            ori_data = pd.read_csv("data/bike-sharing_hourly.csv")
            st.write(ori_data)
            
            
            st.markdown("##### Let's check the datatypes")
            
            dict_schema = {}

            for k in ori_data.columns:
                dict_schema[k] = ori_data[k].dtype
                
            st.write(dict_schema)
            
            st.markdown("### Explore the data by yourself!!")
            
            st.markdown("##### Analysis of `categorical` variable against a `numerical` variable")
            
            numerical_features = ['temp',	'atemp',	'hum',	'windspeed',	'casual',	'registered',	'cnt']
            
            categorical_features = ['season',	'holiday',	'weekday',	'workingday',	'weathersit']
            
            time_series = ['dteday',	'yr',	'mnth',	'hr']
            
            left_column, right_column = st.columns(2)
            
            
            
            with left_column:
                        
                x_selection = st.radio('X_axis', categorical_features)
            
            with right_column:
                
                y_selection = st.radio('Y_axis', numerical_features)
            
            
            
            fig = px.box(data, x=x_selection, y=y_selection, height = 600, width= 800)
            
            st.plotly_chart(fig, use_container_width=True)
                
                
            st.markdown("##### Analysis of two `numerical` variables")    
            
            left_column, right_column = st.columns(2)
            
            
            
            with left_column:
                        
                x_selection2 = st.radio('x_axis', numerical_features)
            
            with right_column:
                
                y_selection2 = st.radio('y_axis', numerical_features)
                
            if x_selection2 == y_selection2:
                
                st.info("Please choose two different features")
                
            else:
                
                fig = px.scatter(data, x=x_selection2, y=y_selection2, height = 600, width= 800)
            
                st.plotly_chart(fig, use_container_width=True)
                
            st.markdown("##### Analysis of `time series`")  
            
            left_column, right_column = st.columns(2)
            
            
            
            with left_column:
                        
                x_selection3 = st.radio('Granularity', time_series)
            
            with right_column:
                
                y_selection3 = st.radio('Y_axis(*) ', numerical_features)
            
            numeric_mean = ['temp',	'atemp', 'hum',	'windspeed']
            numeric_sum = ['casual',	'registered',	'cnt']
            
            if y_selection3 in numeric_mean: 
                
                df_agg = data.groupby(by=x_selection3).mean().reset_index()
                
                #We can include the name of each 
                fig = px.line(df_agg, x=x_selection3, y=y_selection3, height = 600, width= 800)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                df_agg = data.groupby(by=x_selection3).sum().reset_index()
                
                #We can include the name of each 
                fig = px.line(df_agg, x=x_selection3, y=y_selection3, height = 600, width= 800)
                st.plotly_chart(fig, use_container_width=True)
                
            st.markdown("###### (*)The aggregation for `casual`, `registered` and `cnt` is the sum, for the rest is the mean.")
            
            
            
    elif selected == "Insights":
            
            st.title(f"Business {selected}")
            
            lottie_insights = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_22mjkcbb.json")
            
            left_column, right_column = st.columns(2)
            
            
              
            with left_column:
                st.markdown("In this section we will analyze the relationship between the variables that are present on the dataset in order to extract some interesting insights that can help us to understand how the citizens are using the service to know if some changes must be made to optimize costs or provide a better service.") 

            with right_column:
            
                st_lottie(lottie_insights, height=250, key = "business people")
                
            df_daily = data.groupby(by = 'dteday').sum().reset_index()
    
            #from_date = df_daily.dteday.unique()
            #end_date = df_daily.dteday.unique()

            st.markdown("### Business insight 1: users per day")

            start_date, end_date = st.select_slider(
            'Select a date range to visualize',
            options=df_daily.dteday,
            value=('2011-01-01', '2012-12-31'))

            st.write('You selected between', start_date, 'and', end_date)


            df_daily['MA20_casual'] = df_daily.casual.rolling(20).mean()
            df_daily['MA20_registered'] = df_daily.registered.rolling(20).mean()

            ## PLOT FIGURE 1 ##
            df_daily = df_daily[(df_daily['dteday'] >= start_date) &  (df_daily['dteday'] <= end_date)]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_daily.dteday, y=df_daily['registered'],
                            mode='lines',
                            name='registered users', opacity = 0.5))
            fig.add_trace(go.Scatter(x=df_daily.dteday, y=df_daily['casual'],
                            mode='lines',
                            name='casual users', opacity = 0.5))

            fig.add_trace(go.Scatter(x=df_daily.dteday, y=df_daily['MA20_casual'], line_color = 'red',
                            mode='lines', name='casual MA',showlegend = False))
            fig.add_trace(go.Scatter(x=df_daily.dteday, y=df_daily['MA20_registered'],line_color='blue',
                                 mode='lines',showlegend = False)
                     )

            fig.update_yaxes(title_text='Number of users')


            st.plotly_chart(fig, use_container_width=True)

            
            
            lottie_bike2 = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_xz7jx49d.json")
            
            left_column, right_column = st.columns(2)
            
            with left_column:
            
                st.markdown("""
                **Graph insight**: Service is more used by registered users than casual ones. Seasonality can be seen in both groups with peaks in summer/spring and valleys especially during winter.  

                **Recommendation to the business:** Try to promote & increase usage among casuals.
                            """) 

            with right_column:
                            
                st_lottie(lottie_bike2, height=250, key = "bike users")
            
            
            st.markdown("#### Going further from our first inside")
            
            st.markdown("""
                We will study the distribution of total users per month
                            """) 
            
            aggr = data.groupby(by = ['mnth'], as_index=False).sum()
            season_names = ['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer', 'Summer','Summer', 'Fall', 'Fall', 'Fall', 'Winter']
            aggr['season_names'] = season_names
            
            fig = px.bar(aggr, x='mnth', y='cnt', color='season_names')

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
                        **Graph insight**: The bikes are the most used from June until September with 300k+ trips per month and the least used in winter with less than 200k trips per month in January and February.
                        
**Recommendation to the business**: We recommend the business to organise yearly maintenance detailed checks in January and February in order to disturb the least possible network of bikes and stations. 
            """)
            
            
            
            
            st.markdown("### Business insight 2 & 3: behavior of users per hour")

            st.markdown("We will study the use of bikes per hour, and try to identify differences between labor days and weekends")


            options = st.multiselect(
            'Choose the type of day you want to analyze',
            ['Labor days', 'Weekends'])

            data.set_index('dteday', inplace=True)

            data['DayofWeek'] = data['weekday'].replace({1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 0: 'Sunday'})


            Day_of_week = ['Monday','Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            for i in Day_of_week:
                globals()['data_' + str(i)] = data[data['DayofWeek']==i].groupby(by='hr').mean().reset_index()
                globals()['data_' + str(i)]['hr_str'] = globals()['data_' + str(i)].hr.astype(str) + ':00:00'

            if options == ['Labor days']:

                fig = make_subplots(rows=2, cols=3,subplot_titles=("Monday", "Tuesday", "Wedenesday", "Thursday", "Friday"))

                fig.add_trace(
                    go.Scatter(x=data_Monday.hr, y=data_Monday['registered'], name = 'registered' ,showlegend = True, text = "Monday", line_color='blue'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=data_Monday.hr, y=data_Monday['casual'], name = 'casual' ,showlegend = True, text = "Monday", line_color='red'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=data_Tuesday.hr, y=data_Tuesday['registered'], showlegend = False, line_color='blue'),
                    row=1, col=2
                )

                fig.add_trace(
                    go.Scatter(x=data_Tuesday.hr, y=data_Tuesday['casual'], showlegend = False, line_color='red'),
                    row=1, col=2
                )

                fig.add_trace(
                    go.Scatter(x=data_Wednesday.hr, y=data_Wednesday['registered'], showlegend = False, line_color='blue'),
                    row=1, col=3
                )

                fig.add_trace(
                    go.Scatter(x=data_Wednesday.hr, y=data_Wednesday['casual'], showlegend = False, line_color='red'),
                    row=1, col=3
                )


                fig.add_trace(
                    go.Scatter(x=data_Thursday.hr, y=data_Thursday['registered'], showlegend = False, line_color='blue'),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Scatter(x=data_Thursday.hr, y=data_Thursday['casual'], showlegend = False, line_color='red'),
                    row=2, col=1
                )


                fig.add_trace(
                    go.Scatter(x=data_Friday.hr, y=data_Friday['registered'], showlegend = False, line_color='blue'),
                    row=2, col=2
                )

                fig.add_trace(
                    go.Scatter(x=data_Friday.hr, y=data_Friday['casual'], showlegend = False, line_color='red'),
                    row=2, col=2
                )       

                fig.update_layout(height=600, width=800, title_text="Registered and Casual users per hour")


                st.plotly_chart(fig, use_container_width=True)

            elif options == ['Weekends']:

                fig = make_subplots(rows=2, cols=1,subplot_titles=("Saturday", "Sunday"))


                fig.add_trace(
                    go.Scatter(x=data_Saturday.hr, y=data_Saturday['registered'],  name = 'registered' ,showlegend = True, line_color='blue'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=data_Saturday.hr, y=data_Saturday['casual'],  name = 'casual' ,showlegend = True, line_color='red'),
                    row=1, col=1
                )


                fig.add_trace(
                    go.Scatter(x=data_Sunday.hr, y=data_Sunday['registered'], showlegend = False, line_color='blue'),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Scatter(x=data_Sunday.hr, y=data_Sunday['casual'], showlegend = False, line_color='red'),
                    row=2, col=1
                )

                fig.update_layout(height=600, width=800, title_text="Registered and Casual users per hour")


                st.plotly_chart(fig, use_container_width=True)


            elif (options == ['Labor days','Weekends']) or (options == ['Weekends','Labor days']):

                fig = make_subplots(rows=3, cols=3,subplot_titles=("Monday", "Tuesday", "Wedenesday", "Thursday", "Friday", "Saturday", "Sunday"))

                fig.add_trace(
                    go.Scatter(x=data_Monday.hr, y=data_Monday['registered'],  name = 'registered' ,showlegend = True, text = "Monday", line_color='blue'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=data_Monday.hr, y=data_Monday['casual'],  name = 'casual' ,showlegend = True, text = "Monday", line_color='red'),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(x=data_Tuesday.hr, y=data_Tuesday['registered'], showlegend = False, line_color='blue'),
                    row=1, col=2
                )

                fig.add_trace(
                    go.Scatter(x=data_Tuesday.hr, y=data_Tuesday['casual'], showlegend = False, line_color='red'),
                    row=1, col=2
                )

                fig.add_trace(
                    go.Scatter(x=data_Wednesday.hr, y=data_Wednesday['registered'], showlegend = False, line_color='blue'),
                    row=1, col=3
                )

                fig.add_trace(
                    go.Scatter(x=data_Wednesday.hr, y=data_Wednesday['casual'], showlegend = False, line_color='red'),
                    row=1, col=3
                )


                fig.add_trace(
                    go.Scatter(x=data_Thursday.hr, y=data_Thursday['registered'], showlegend = False, line_color='blue'),
                    row=2, col=1
                )

                fig.add_trace(
                    go.Scatter(x=data_Thursday.hr, y=data_Thursday['casual'], showlegend = False, line_color='red'),
                    row=2, col=1
                )


                fig.add_trace(
                    go.Scatter(x=data_Friday.hr, y=data_Friday['registered'], showlegend = False, line_color='blue'),
                    row=2, col=2
                )

                fig.add_trace(
                    go.Scatter(x=data_Friday.hr, y=data_Friday['casual'], showlegend = False, line_color='red'),
                    row=2, col=2
                )


                fig.add_trace(
                    go.Scatter(x=data_Saturday.hr, y=data_Saturday['registered'], showlegend = False, line_color='blue'),
                    row=2, col=3
                )

                fig.add_trace(
                    go.Scatter(x=data_Saturday.hr, y=data_Saturday['casual'], showlegend = False, line_color='red'),
                    row=2, col=3
                )


                fig.add_trace(
                    go.Scatter(x=data_Sunday.hr, y=data_Sunday['registered'], showlegend = False, line_color='blue'),
                    row=3, col=1
                )

                fig.add_trace(
                    go.Scatter(x=data_Sunday.hr, y=data_Sunday['casual'], showlegend = False, line_color='red'),
                    row=3, col=1
                )


                fig.update_layout(height=600, width=800, title_text="Registered and Casual users per hour")


                st.plotly_chart(fig, use_container_width=True)

            else:

                st.info("None of the options were selected")

#------------------Lottie animation----------------------------

            lottie_bikes = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_kzfpndct.json")
    
            
            left_column, right_column = st.columns(2)
            
            with left_column:
        
                st.markdown("**Graph insights**")
                st.markdown("""
                1. The customers’ behaviour significantly differ depending on whether they are casual or registered users and whether it is a weekday or a weekend.  
- Casual users have a relatively low and flat usage of bikes during the weekdays (<100 trips on average per h) and a higher usage on the weekend peaking in the early afternoon with raround 150 trips per hour.  
                
- Registered users have two rush periods during the weekdays with more than 400 trips per hour: at 8am and from 5 until 6pm (except on Fridays where it only peaks at 5pm). On weekends, they have a similar usage pattern as the casual users peaking at 12pm (>200 trips per hour) with a relatively flatter curve.

2. Both casual and registered users almost do not use the bikes at all from 11pm until 5am except for Friday and Saturday nights for registered users (between 30-100 trips per hour from 11pm until 3am).
                """)
            with right_column:
                
                st_lottie(lottie_bikes, height=700, key = "bikes on a city")
                

            left_column, right_column = st.columns(2)
            
            with right_column:
                
                st.markdown("**Recomendations to business**")
                st.markdown("""
               1. Happy hours: In order to incentivize potential casual users to use the bike while going back from a drink or a restaurant, the company could implement happy hours from 11pm until 5am making the price of one bike trip cheaper during these hours at night.  
               
2. Partner with companies: It seems a large part of the registered users are users that go to work biking. We recommend exploring the option of partnering with large companies present in the city center to incentivize their employees to use the bike to go to the office.

                """)
            
            with left_column:
                lottie_iot = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_UExBCF.json")
                st_lottie(lottie_iot, height=400, key = "city")                            
                                            
                                            
                                            
                                            
                                            
            st.markdown("### Additional insights")                                
            st.markdown("##### Let's study the climate indicators")  
            
            selected = option_menu(menu_title = None,
                               options = ["Weather situation", "Humidity", "Wind speed"],
                               icons = ["cloud","droplet-half","wind"],
                               menu_icon = "cast",
                               default_index = 0,
                               orientation = "horizontal"    
                              )
        
            if selected == "Weather situation":
               
                st.markdown("###### **Distribution of total users per weather situation**")
                
                
            
                fig = px.box(data, x="weathersit", y="cnt", height = 500, width= 800)
                st.plotly_chart(fig, use_container_width=True)

                left_column, right_column = st.columns(2)
                lottie_weather = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_wuqUXi.json")
                with left_column:
                    
                    st_lottie(lottie_weather, height=300, key = "sun and rain")
                
                with right_column:
                    
                    st.markdown(""" 
                                We noticed that the worse the weather situation, people tend to avoid using bikes as a mean of transport for several reasons:  

                                - In rainy environment, users are completely discouraged to use the bikes.
                                - The cold does not help for using an open mean of transport. Casual users are more susceptible to adverse temperatures.

                                """) 
                    
            if selected == "Humidity":
                
                st.markdown("###### **Distribution of total users per humidity level**")
                aggr_hum = data.groupby('hum', as_index=False).sum()
                fig = px.bar(aggr_hum, x='hum', y='cnt')
                fig.update_layout(
                    xaxis_title = 'Humidity',
                    yaxis_title ='Number of users'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                left_column, right_column = st.columns(2)
                lottie_hum = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_74gqrlus.json")
                
                with left_column:
                    
                    st_lottie(lottie_hum, height=100, key = "humidity")
                
                with right_column:
                    
                    st.markdown(""" 
                                Humidity follows a trend present in weather features. This trend is that too high or too low values cause the number of users to drop. On the higher end of the scale there are some outliers but the number of users is relatively smaller compared to the middle of the graph.

                                """) 
            if selected == "Wind speed":
                
                st.markdown("###### **Distribution of total users per wind speed**")
                aggr_wind = data.groupby('windspeed', as_index=False).sum()
                fig = px.bar(aggr_wind, x='windspeed', y='cnt')
                fig.update_layout(
                    xaxis_title = 'Windspeed',
                    yaxis_title ='Number of users'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                left_column, right_column = st.columns(2)
                lottie_wind = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_J8EPQ1.json")
                
                with left_column:
                    
                    st_lottie(lottie_wind, height=100, key = "windspeed")
                
                with right_column:
                    
                    st.markdown(""" 
                                The number of users per wind speed measure follows a similar pattern as other weather indicators. However in this case the only too high values of wind speed cause user decrease is too high, as 0 wind speed means that there is no wind.
                                """) 
                
                
                
                
                
                
                
                
                
                
    elif selected == "ML model":
            st.title(f" Building a predictor of users")
            
            st.markdown(""" 
                        In this section, we will explain the approach we followed to estimate the number of bike users, given the explanatory variables we had on the dataset. 
                        
                        As we have seen, there are clear differences between the two types of users. The registered users tend to use this bikes across the hole week, they use the bikes for going work or school (peak on the laboor days in the morning) and also for returning home (peak on the labor days in the afternoon). While the casuals users do not use the bikes in the same way, they maily use the bikes during weekends throughout the day, apart from that, we have seen that the casual users are much more sensitive to bad weather conditions than the registered users.
                        
                        Based on that we have chosen the following approach in order to build our prediction:
                                """) 
            
            
            st.markdown("We know that:")
            st.latex ("Total\_users = Registered\_users + Casual\_users = f(X)")
            st.markdown("###### (*)Where f(X) is a function of all the explanatory features we already have in the model.")
            
            st.markdown("So in this case, given that we have two variables that added together forms the target we want to predict, and these two variables are sufficiently separable, we planned to treat thes two variables as separate targets and estimate each of them independently.")
            
            st.markdown("This means:")
            st.latex ("Registered\_users = f_1(X)...(1)")
            
            #st.markdown("AND")
            
            st.latex ("Casual\_users = f_2(X)...(2)")
            
            st.latex ("Total\_users = Registered\_users + Casual\_users ...(3)")
            
            st.markdown("""
            After trying different models using `pycaret` library on a different environment, we kept the best performing models for estimating the two types of users per hour.  
            
            The model chosen for predicting both types of users was the `Random Forest Regressor` with different hyperparameters per model.
            """)
            
            st.markdown("""
            #### Casual users predictor 
            """)
            
#Creation of the dataset to build the model.
            def day_time(data):
    
                if data['hr'] >= 6 and data['hr']<=10:
                    val='Morning'
                elif data['hr'] > 10 and data['hr'] < 16: 
                    val='Afternoon'
                elif data['hr'] >= 16 and data['hr'] <=20:
                    val='Evening'
                elif (data['hr'] > 20 and data['hr'] <= 23) or (data['hr']>=0 and data['hr']<6):
                    val='Night'

                return val 
    
            data['day_time'] = data.apply(day_time, axis=1)
        
        
            def ad_hum(data):
  
                if data['hum'] >= 0.2 or data['hum'] <= 0.88:
                    return 1
                else:
                    return 0
                
            data['ad_hum'] = data.apply(ad_hum, axis=1)
            
            def ad_wind(data):
  
                if data['windspeed'] <= 0.42:
                    return 1
                else:
                    return 0
            data['ad_wind'] = data.apply(ad_wind, axis=1)
            
            def ad_temp(data):
  
                if data['temp'] >= 0.2 or data['temp'] <= 0.86:
                    return 1
                else:
                    return 0
            
            data['ad_temp'] = data.apply(ad_temp, axis=1)
        
        
        
            df = data
            df = df.reset_index()
            df.drop(['dteday','instant','holiday', 'DayofWeek', 'casual','registered','Datetime','yr','weekday','hr_str','index'], axis=1, inplace=True)
            df_c = data
            df_c = df_c.reset_index()
            df_c.drop(['dteday','instant','holiday', 'DayofWeek','registered','Datetime','yr','weekday','hr_str','index'], axis=1, inplace=True)
            df_c.drop('cnt',axis=1,inplace=True)
            df_c.drop('atemp',axis=1,inplace=True)
            df_r = data
            df_r = df_r.reset_index()
            df_r.drop(['dteday','instant','holiday', 'DayofWeek','casual','Datetime','yr','weekday','hr_str','index'], axis=1, inplace=True)
            df_r.drop('cnt',axis=1,inplace=True)
            df_r.drop('atemp',axis=1,inplace=True)
            
            
            def day_time_enc(df):
  
                if df['day_time']=='Morning':
                    return 1
                elif df['day_time']=='Afternoon':
                    return 2
                elif df['day_time']=='Evening':
                    return 3
                elif df['day_time']=='Night':
                    return 4
                
            df['day_time'] = df.apply(day_time_enc, axis=1)
            df_c['day_time'] = df_c.apply(day_time_enc, axis=1)
            df_r['day_time'] = df_r.apply(day_time_enc, axis=1)
            
            
# Model for casual users

            y_c = pd.Series(df_c['casual'], name="Target")
            df_c.drop(['casual'], axis=1,inplace=True)
            X_c = pd.DataFrame(df_c)
            X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_c,y_c, test_size=0.25)
            
            
            best_param_casual = {'max_depth': 10,
 'min_samples_leaf': 1,
 'min_samples_split': 3,
 'n_estimators': 300}
            st.markdown("After running the `GridSearchCV` method, we identified that the best parametes for this model are:")
            st.json(best_param_casual)
            
            reg_casual = RandomForestRegressor(
                                                max_depth=10,
                                                min_samples_leaf=1,
                                                min_samples_split=3,
                                                n_estimators= 300
                                                )
            
            reg_casual.fit(X_c_train,y_c_train)
            
            
            acc_reg_casual = reg_casual.score(X_c_test, y_c_test)
            
            y_c_test_pred = reg_casual.predict(X_c_test)
            
            MAEc = mean_absolute_error(y_c_test, y_c_test_pred)
            
            
            st.markdown("##### Results of the model:")
            
            st.write(f"R_squared = {round(acc_reg_casual, 3)}")
            
            st.write(f"MAE = {round(MAEc, 0)}")
            
            
            st.markdown("""
            #### Registered users predictor 
            """)
            
            y_r = pd.Series(df_r['registered'], name="Target")
            df_r.drop(['registered'], axis=1,inplace=True)
            X_r = pd.DataFrame(df_r)
            X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_r,y_r, test_size=0.25)
            
            best_param_registered = {'max_depth': 10,
 'min_samples_leaf': 3,
 'min_samples_split': 2,
 'n_estimators': 200}
            st.markdown("After running the `GridSearchCV` method, we identified that the best parametes for this model are:")
            
            st.json(best_param_registered)
            
            reg_registered = RandomForestRegressor(
        max_depth=10,
        min_samples_leaf=3,
        min_samples_split=2,
        n_estimators= 200)
            
            reg_registered.fit(X_r_train,y_r_train)
            
            acc_reg_registered = reg_registered.score(X_r_test, y_r_test)
            
            y_r_test_pred = reg_registered.predict(X_r_test)
            
            MAEr = mean_absolute_error(y_r_test, y_r_test_pred)
            
            
            st.markdown("##### Results of the model:")
            
            st.write(f"R_squared = {round(acc_reg_registered, 3)}")
            
            st.write(f"MAE = {round(MAEr, 0)}")
            
            
            st.markdown("##### Get your own prediction: Casual Users")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                season = st.text_input(
        "Season",
        "1: Winter, 2: Spring, 3: Summer, 4: Autumn")
                
                workingday = st.text_input(
        "Working day",
        "1: Working day, 0: Otherwise")
                
                hum = st.text_input(
        "Humidity",
        "Integer")
                
                ad_wind = st.text_input(
        "Appropriate windspeed",
        "1: Appropriate, 0: Otherwise")
                
                
            with col2:
                mnth = st.text_input("Month",
            "1: January - 12:December ")

                weathersit = st.text_input(
            "Weather situation",
            "1: Clear, 2: Cloudy, 3: Light rain/snow, 4: Heavy rain/snow")

                windspeed = st.text_input(
            "Windspeed",
            "Integer")

                ad_hum = st.text_input(
            "Appropriate humidity",
            "1: Appropriate, 0: Otherwise")
                
            with col3:
                hr = st.text_input(
            "Hour",
            "From 0 to 24")

                temp = st.text_input(
            "Temperature",
            "Integer")

                day_time = st.text_input(
            "Day_time",
            "1: Morning, 2: Afternoon, 3: Evening, 4: Night")

                ad_temp = st.text_input(
            "Appropriate Temperature",
            "1: Appropriate, 0: Otherwise")
                
                
            
            
            try:
                
                pred_c = [{
    "season": int(season),
    "mnth": int(mnth),
    "hr": int(hr),
    "workingday": int(workingday),
    "weathersit": int(weathersit),
    "temp": int(temp),
    "hum": int(hum),
    "windspeed": int(windspeed),
    "day_time": int(day_time),
    "ad_hum": int(ad_hum),
    "ad_wind": int(ad_wind),
    "ad_temp": int(ad_temp),
}]    
                pred_c = pd.DataFrame.from_dict(pred_c, orient="columns")
                #st.write(X_c_train)
                #st.write(pred_c)
                
                prediction = np.rint(reg_casual.predict(pred_c))
                prediction_reg = np.rint(reg_registered.predict(pred_c),0)
                st.markdown("**Your prediction for casual users is:**") 
                st.write(prediction)
                
                
                st.markdown("**Your prediction for registered users is:**") 
                st.write(prediction_reg)
                
                st.markdown("**Your prediction for total users is:**") 
                st.write(prediction + prediction_reg)
            except:
                st.info("Please include the proper datatype according to the information on boxes")
                
                
            
            
            
if __name__ == "__main__":

    # This is to configure some aspects of the app
    st.set_page_config(layout="wide", page_title="Bike sharing Washington", page_icon="🚴‍♂️")

    # Write titles in the main frame and the side bar
    
   

    # Call main function
    main()