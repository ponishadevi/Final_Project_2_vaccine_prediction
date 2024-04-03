import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
import plotly.express as px
import pickle
# Set page title and favicon
st.set_page_config(page_title="Vaccine Usage Analysis and Prediction", page_icon=":syringe:")
# Load the dataset
@st.cache_data
def load_data(url):
    return pd.read_csv(url)

def preprocess_data(data, decimal_precision=2):
    # Impute missing values with mean for numerical columns
    numerical_imputer = SimpleImputer(strategy='mean')
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])
    
    # Impute missing values with most frequent for categorical columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
    
    # Round numerical features to specified decimal precision
    data = data.round(decimals=decimal_precision)
    
    return data

# Preprocess the data
url = "https://raw.githubusercontent.com/nethajinirmal13/Training-datasets/main/Vaccine.csv"
vac_data = load_data(url)
vac_data = preprocess_data(vac_data)

# Sidebar menu
with st.sidebar:
    # Use st.selectbox to create main menu with custom styling
    select = st.selectbox("Main Menu", ["Home", "Data Exploration", "Model Predict"], format_func=lambda x: f'{x}', key="custom_selectbox")
         


# Dashboard pages
if select == "Home":
    st.title("Welcome to the Vaccine Usage Analysis and Prediction Dashboard")
    st.image("h1n1_5.jpg", use_column_width=True)
    


    # Header
    
    st.markdown("---")
    st.markdown("## Welcome!")
    st.write("Welcome to the Vaccine Usage Analysis and Prediction Dashboard! This dashboard aims to provide insights into individuals' likelihood of receiving the H1N1 flu vaccine and to facilitate effective targeting of vaccination campaigns.")

    # Project Overview
    st.markdown("---")
    st.markdown("## Project Overview")
    st.write("**Project Title:** Vaccine Usage Analysis and Prediction")
    st.write("**Skills Takeaway:** Python scripting, Pandas, Data Visualization, and Machine Learning")
    st.write("**Domain:** Data analysis and Machine Learning")

    # Add additional lines for sidebar content
    st.sidebar.header("About")
    st.sidebar.write("The aim of this study is to predict how likely individuals are to receive their H1N1 flu vaccine.")
    st.sidebar.write("We believe the prediction outputs (model and analysis) of this study will give public health professionals and policymakers, as an end user, a clear understanding of factors associated with low vaccination rates.")
    st.sidebar.write("This in turn, enables end users to systematically act on those features hindering people to get vaccinated.")
    st.sidebar.write("In this project, we used the survey data from the National Flu Survey (NHFS, 2009) to predict whether or not respondents got the H1N1 vaccine.")

        # Dataset Information
    st.markdown("---")
    st.markdown("## Dataset Information")
    st.write("The dataset contains various attributes related to respondents' behaviors, perceptions, and demographics. Some of the key attributes include:")
    st.write("- **h1n1_worry**: Worry about the H1N1 flu")
    st.write("- **h1n1_awareness**: Level of awareness about the H1N1 flu")
    st.write("- **antiviral_medication**: Whether the respondent has taken antiviral medication")
    st.write("- **contact_avoidance**: Whether the respondent has avoided close contact with people who have flu-like symptoms")
    st.write("- **bought_face_mask**: Whether the respondent has bought a mask")
    st.write("- **wash_hands_frequently**: Whether the respondent washes hands frequently or uses hand sanitizer")
    st.write("And many more...")

    # How to Use this Dashboard
    st.markdown("---")
    st.markdown("## How to Use this Dashboard")
    st.write("Navigate through the sidebar menu to explore different sections of the dashboard:")
    st.write("- **Home**: You are currently on the home page.")
    st.write("- **Data Exploration**: Explore and visualize the dataset.")
    st.write("- **Top Charts**: View top charts and insights.")
    st.write("- **Model Development**: Train and evaluate machine learning models.")
    st.write("- **Prediction**: Make predictions using the trained model.")

    # Next Steps
    st.markdown("---")
    st.markdown("## Next Steps")
    st.write("Feel free to explore the different sections of the dashboard to gain insights into vaccine usage patterns and predictions. If you have any questions or feedback, please don't hesitate to reach out.")

    # Let's Get Started
    st.markdown("---")
    st.markdown("## Let's Get Started!")
    st.write("Click on the sidebar menu to explore the dashboard features and gain valuable insights.")
elif select == "Data Exploration":
    # Add data exploration content here
    # Grouped bar plot for Seasonal Vaccine Recommendations
    st.title("Grouped Bar Plot for Seasonal Vaccine Recommendations")
    grouped_data = vac_data.groupby(['dr_recc_seasonal_vacc']).agg({'is_seas_vacc_effective': 'count',
                                                                'is_seas_risky': 'count',
                                                                'sick_from_seas_vacc': 'count'})
    fig, ax = plt.subplots(figsize=[8, 6])
    grouped_data.plot(kind='bar', cmap='tab20b', ax=ax)
    plt.ylabel('Count')
    plt.title('Grouped Bar Plot for Seasonal Vaccine Recommendations')
    plt.xticks(rotation=0)
    plt.legend(title='Attributes')
    st.pyplot(fig)

    # Grouped bar plot for H1N1 Vaccine Recommendations
    st.title("Grouped Bar Plot for H1N1 Vaccine Recommendations")
    grouped_data1 = vac_data.groupby(['dr_recc_h1n1_vacc']).agg({'is_h1n1_vacc_effective': 'count',
                                                                'is_h1n1_risky': 'count',
                                                                'sick_from_h1n1_vacc': 'count'})
    fig, ax = plt.subplots(figsize=[8, 6])
    grouped_data1.plot(kind='bar', cmap='tab20c', ax=ax)
    plt.ylabel('Count')
    plt.title('Grouped Bar Plot for H1N1 Vaccine Recommendations')
    plt.xticks(rotation=0)
    plt.legend(title='Attributes')
    st.pyplot(fig)

    # Countplot for Doctor Recommendations
    st.title("Countplot for Doctor Recommendations")
    fig, ax = plt.subplots(1, 2, figsize=[10, 6], sharey=True)
    sns.countplot(x='dr_recc_h1n1_vacc', data=vac_data, color='maroon', ax=ax[0])
    ax[0].set_title('Doctor Recommended H1N1 Vaccine')
    sns.countplot(x='dr_recc_seasonal_vacc', data=vac_data, color='navy', ax=ax[1])
    ax[1].set_title('Doctor Recommended Seasonal Vaccine')
    st.pyplot(fig)
   

 

    # Sidebar menu
    with st.sidebar:
        st.title("Filter Options")
        age_filter = st.multiselect("Filter by Age Bracket", vac_data['age_bracket'].unique())
        h1n1_worry_filter = st.multiselect("Filter by H1N1 Worry", vac_data['h1n1_worry'].unique())
        h1n1_awareness_filter = st.multiselect("Filter by H1N1 Awareness", vac_data['h1n1_awareness'].unique())

    # Main content
    st.title("Perceived Risk of Getting Sick vs. Vaccine Uptake")

    # Filter data based on selected filters
    subset_data = vac_data.copy()  # Make a copy of the original data
    if age_filter:
        subset_data = subset_data[subset_data['age_bracket'].isin(age_filter)]
    if h1n1_worry_filter:
        subset_data = subset_data[subset_data['h1n1_worry'].isin(h1n1_worry_filter)]
    if h1n1_awareness_filter:
        subset_data = subset_data[subset_data['h1n1_awareness'].isin(h1n1_awareness_filter)]

    # Create a scatter plot using Plotly
    fig = px.scatter(subset_data, x='sick_from_h1n1_vacc', y='h1n1_vaccine', color='h1n1_vaccine',
                    labels={'sick_from_h1n1_vacc': "Perceived Risk of Getting Sick from H1N1 without Vaccine",
                            'h1n1_vaccine': "H1N1 Vaccine Uptake"},
                    title="Perceived Risk of Getting Sick from H1N1 without Vaccine vs. H1N1 Vaccine Uptake",
                    color_discrete_map={'0': 'blue', '1': 'green'})
    fig.update_traces(marker=dict(size=12), selector=dict(mode='markers'))

    # Add sick from seasonal vaccine to the scatter plot
    fig.add_scatter(x=subset_data['sick_from_seas_vacc'], y=subset_data['h1n1_vaccine'],
                    mode='markers', name='Sick from Seasonal Vaccine', marker=dict(color='red', size=12))

    # Update layout
    fig.update_layout(legend_title="H1N1 Vaccine",
                    legend=dict(itemsizing='constant'),
                    xaxis=dict(title="Perceived Risk of Getting Sick from H1N1 without Vaccine"),
                    yaxis=dict(title="H1N1 Vaccine Uptake"))

    # Render the Plotly figure
    st.plotly_chart(fig)




    # Combine contact avoidance variables into a single feature
    vac_data['contact_avoidance'] = vac_data[['avoid_large_gatherings', 'reduced_outside_home_cont', 'avoid_touch_face', 'chronic_medic_condition']].sum(axis=1)

    # Map the combined feature to categorical levels
    contact_avoidance_mapping = {
        0: 'Low',
        1: 'Moderate',
        2: 'High',
        3: 'Very High',
        4: 'Extreme'
    }
    vac_data['contact_avoidance'] = vac_data['contact_avoidance'].map(contact_avoidance_mapping)

    # Group data based on combined contact avoidance and H1N1 vaccine uptake
    grouped_data = vac_data.groupby(['contact_avoidance', 'h1n1_vaccine']).size().unstack(fill_value=0)

    # Calculate the percentage of individuals who took the H1N1 vaccine (yes) and who didn't (no) for each level of contact avoidance
    grouped_data_percentage = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_data_percentage.plot(kind='bar', stacked=True, color=['orange', 'lightblue'], ax=ax)  # Custom colors
    ax.set_xlabel('Contact Avoidance Level')
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage of H1N1 Vaccine Uptake by Contact Avoidance Level')
    ax.legend(title='H1N1 Vaccine', loc='upper right')

    # Display the plot using st.pyplot()
    st.pyplot(fig)



    # Distribution of H1N1 Worry based on H1N1 Vaccine
    st.subheader("Distribution of H1N1 Worry based on H1N1 Vaccine")
    h1n1_worry_counts_vaccine = vac_data.groupby('h1n1_vaccine')['h1n1_worry'].value_counts().unstack()
    st.bar_chart(h1n1_worry_counts_vaccine)

    # Distribution of H1N1 Awareness
    st.subheader("Distribution of H1N1 Awareness")
    h1n1_awareness_counts = vac_data['h1n1_awareness'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(h1n1_awareness_counts, labels=h1n1_awareness_counts.index, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)



    # Vaccine acceptance/non-acceptance by age bracket
    st.title("Vaccine Acceptance/Non-acceptance by Age Bracket")
    age_acceptance_counts = vac_data.groupby(['age_bracket', 'h1n1_vaccine']).size().unstack(fill_value=0)
    st.bar_chart(age_acceptance_counts, color=["#1f77b4", "#ff7f0e"])  # Blue and Orange

        # Vaccine acceptance/non-acceptance by education level
    st.title("Vaccine Acceptance/Non-acceptance by Education Level")
    education_acceptance_counts = vac_data.groupby(['qualification', 'h1n1_vaccine']).size().unstack(fill_value=0)
    st.bar_chart(education_acceptance_counts, color=["#3366FF", "#FF3366"])  # Custom colors

    # Vaccine acceptance/non-acceptance by income level
    st.title("Vaccine Acceptance/Non-acceptance by Income Level")
    income_acceptance_counts = vac_data.groupby(['income_level', 'h1n1_vaccine']).size().unstack(fill_value=0)
    st.line_chart(income_acceptance_counts, use_container_width=True, color=["#FF6633", "#33FFCC"])  # Custom colors





      # Main content
    st.title("Vaccine Acceptance/Non-acceptance by H1N1 Awareness and Worry")

    # Filter data based on selected filters
    filtered_data = vac_data[
        (vac_data['h1n1_worry'].isin(h1n1_worry_filter)) &
        (vac_data['h1n1_awareness'].isin(h1n1_awareness_filter))
    ]

    # Group data based on H1N1 awareness and worry levels
    grouped_data = filtered_data.groupby(['h1n1_worry', 'h1n1_awareness', 'h1n1_vaccine']).size().unstack(fill_value=0)

    # Calculate the percentage of individuals who took the H1N1 vaccine (yes) and who didn't (no) for each combination of H1N1 awareness and worry levels
    grouped_data_percentage = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100

    # Reset index to make columns accessible
    grouped_data_percentage.reset_index(inplace=True)

    # Melt the DataFrame to prepare for plotting
    melted_data = grouped_data_percentage.melt(id_vars=['h1n1_worry', 'h1n1_awareness'], var_name='H1N1 Vaccine', value_name='Percentage')

    # Plot the results using Plotly
    fig = px.bar(melted_data, x='h1n1_worry', y='Percentage', color='h1n1_awareness', barmode='group',
                labels={'h1n1_worry': 'H1N1 Worry Levels', 'Percentage': 'Percentage of Vaccine Acceptance/Non-acceptance'},
                title='Vaccine Acceptance/Non-acceptance by H1N1 Awareness and Worry',
                color_discrete_map={'No': 'blue', 'Little': 'green', 'Good': 'red'})  # Adjust color mapping as needed
    fig.update_layout(legend_title='H1N1 Awareness', xaxis_title='H1N1 Worry Levels', yaxis_title='Percentage')
    st.plotly_chart(fig)



    # Define colors for each h1n1_vaccine status
    status_colors = {
        0: 'rgb(255, 99, 71)',   # Non-acceptance (red)
        1: 'rgb(65, 105, 225)'    # Acceptance (royal blue)
    }

    # Distribution of Number of Adults and Children based on H1N1 Vaccine Acceptance
    st.title("Distribution of Number of Adults and Children based on H1N1 Vaccine Acceptance")

    # Calculate counts for adults and children for each h1n1_vaccine status
    adults_counts = vac_data.groupby(['h1n1_vaccine', 'no_of_adults']).size().unstack(fill_value=0)
    children_counts = vac_data.groupby(['h1n1_vaccine', 'no_of_children']).size().unstack(fill_value=0)

    # Create subplots
    fig = go.Figure()

    # Add bar chart for adults
    for status, color in status_colors.items():
        fig.add_trace(go.Bar(x=adults_counts.index, y=adults_counts[status],
                            name=f"Adults - {'Acceptance' if status == 1 else 'Non-acceptance'}",
                            marker_color=color))

    # Add bar chart for children
    for status, color in status_colors.items():
        fig.add_trace(go.Bar(x=children_counts.index, y=children_counts[status],
                            name=f"Children - {'Acceptance' if status == 1 else 'Non-acceptance'}",
                            marker_color=color))

    # Update layout
    fig.update_layout(barmode='group', xaxis_title='Number of People', yaxis_title='Count',
                    title='Distribution of Number of Adults and Children based on H1N1 Vaccine Acceptance')

    # Display the combined bar chart
   
   
    st.plotly_chart(fig)





# Main content
    st.title("Doctor's Recommendation vs. Vaccine Uptake")

    # Filter data based on selected filters
    filtered_data = vac_data.copy()  # Make a copy of the original data
    if age_filter:
        filtered_data = filtered_data[filtered_data['age_bracket'].isin(age_filter)]
    if h1n1_worry_filter:
        filtered_data = filtered_data[filtered_data['h1n1_worry'].isin(h1n1_worry_filter)]
    if h1n1_awareness_filter:
        filtered_data = filtered_data[filtered_data['h1n1_awareness'].isin(h1n1_awareness_filter)]

    # Calculate doctor's recommendation for H1N1 vaccine
    doctor_recommendation_h1n1 = filtered_data.groupby('dr_recc_h1n1_vacc')['h1n1_vaccine'].mean() * 100

    # Calculate doctor's recommendation for seasonal vaccine
    doctor_recommendation_seasonal = filtered_data.groupby('dr_recc_seasonal_vacc')['h1n1_vaccine'].mean() * 100

    # Create a DataFrame for both recommendations
    combined_data = pd.DataFrame({
        "Doctor's Recommendation for H1N1 Vaccine": doctor_recommendation_h1n1,
        "Doctor's Recommendation for Seasonal Vaccine": doctor_recommendation_seasonal
    })

    # Create a plotly figure
    fig = go.Figure()

    # Add bar traces for each recommendation
    fig.add_trace(go.Bar(x=combined_data.index, y=combined_data["Doctor's Recommendation for H1N1 Vaccine"], name="H1N1 Vaccine", marker_color='blue'))
    fig.add_trace(go.Bar(x=combined_data.index, y=combined_data["Doctor's Recommendation for Seasonal Vaccine"], name="Seasonal Vaccine", marker_color='green'))

    # Update layout
    fig.update_layout(title="Doctor's Recommendation vs. Vaccine Uptake",
                    xaxis_title="Doctor's Recommendation",
                    yaxis_title="Vaccine Uptake",
                    barmode='group')

    # Render the plotly figure
    st.plotly_chart(fig)

 
        
elif select == "Model Predict":
    # Add the HTML styling
    Body_html = """
    <style>
    h1 {
        color: #7a7477;
    }

    h2 {
        color: #F3005E;
        margin: 0;
        position: absolute;
        top: 50%;
        left: 50%;
        margin-right: -50%;
        transform: translate(-50%, -50%)
    }

    h3 {
        color: #898989;
        display: block;
        margin-left: auto;
        margin-right: auto;
        size: 200%;
    }

    h4 {
        color: #7a7477;
    }

    h5 {
        color: lightgrey;
    }

    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        height: auto;
    }
    body {
        background-image: url(https://i.stack.imgur.com/HCfU2.png);
    }
    </style>
    """
    st.markdown(Body_html, unsafe_allow_html=True)  # Body rendering

    # Header content
    st.write(
        """
    # Will a Person get the H1N1 Vaccine?

    ![vaccine_img](http://bento.cdn.pbs.org/hostedbento-prod/filer_public/spillover/images/viruses/definitions/vaccines.png)

    #### Try the tool to predict whether a person will get the H1N1 vaccine using the information they shared about their healthcare background and opinions.

    ***

    In Spring 2009, a pandemic caused by the H1N1 influenza virus ("Swine Flu"), swept across the world. A vaccine for the H1N1 flu virus became publicly available in October 2009.
    In late 2009 and early 2010, the United States conducted the National 2009 H1N1 Flu Survey. More details about this dataset and features are available at [DrivenData.org](https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/).

    ***

    """
    )

    # Function to preprocess input data
    def preprocess_input(age_bracket, h1n1_worry, h1n1_awareness, dr_recc_h1n1_vacc, is_h1n1_vacc_effective, is_h1n1_risky):
        # Map input values to numerical values expected by the model
        age_mapping = {"18 - 34 Years": 0, "35 - 44 Years": 1, "45 - 54 Years": 2, "55 - 64 Years": 3, "64+ Years": 4}
        h1n1_mapping = {"Not at all": 0, "A little": 1, "Somewhat": 2, "Very": 3, "Extremely": 4}
        doctor_mapping = {"Yes": 1, "No": 0, "Don't know": 2}
        
        # Convert input values to numerical format
        age_numeric = age_mapping[age_bracket]
        h1n1_worry_numeric = h1n1_mapping[h1n1_worry]
        h1n1_awareness_numeric = h1n1_mapping[h1n1_awareness]
        dr_recc_h1n1_vacc_numeric = doctor_mapping[dr_recc_h1n1_vacc]
        is_h1n1_vacc_effective_numeric = h1n1_mapping[is_h1n1_vacc_effective]
        is_h1n1_risky_numeric = h1n1_mapping[is_h1n1_risky]
        
        # Return a list of all 31 features (some with dummy values)
        return [age_numeric, h1n1_worry_numeric, h1n1_awareness_numeric, dr_recc_h1n1_vacc_numeric, is_h1n1_vacc_effective_numeric, is_h1n1_risky_numeric] + [0] * 25

    # Load the trained model
    with open('DecisionTreeClassifier.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    # Streamlit interface
    st.title("Predict H1N1 Vaccine Uptake")

    # Input fields for features
    age_bracket = st.selectbox("Age Bracket", ["18 - 34 Years", "35 - 44 Years", "45 - 54 Years", "55 - 64 Years", "64+ Years"])
    h1n1_worry = st.selectbox("Worry about H1N1 Flu", ["Not at all", "A little", "Somewhat", "Very", "Extremely"])
    h1n1_awareness = st.selectbox("Awareness of H1N1 Flu", ["Not at all", "A little", "Somewhat", "Very"])
    dr_recc_h1n1_vacc = st.selectbox("Doctor Recommended H1N1 Vaccine", ["Yes", "No", "Don't know"])
    is_h1n1_vacc_effective = st.selectbox("Is H1N1 Vaccine Effective?", ["Not at all", "A little", "Somewhat", "Very", "Extremely"])
    is_h1n1_risky = st.selectbox("Is H1N1 Flu Risky?", ["Not at all", "A little", "Somewhat", "Very", "Extremely"])

    # Button to trigger prediction
    if st.button("Predict"):
        # Preprocess input data
        input_data = preprocess_input(age_bracket, h1n1_worry, h1n1_awareness, dr_recc_h1n1_vacc, is_h1n1_vacc_effective, is_h1n1_risky)
        
        # Make prediction
        prediction = loaded_model.predict([input_data])
        
        # Display prediction result
        if prediction[0] == 1:
            st.success("The individual is likely to get the H1N1 vaccine.")
        else:
            st.error("The individual is unlikely to get the H1N1 vaccine.")



    feature_importance = {
        "dr_recc_h1n1_vacc": 0.14,
        "is_h1n1_vacc_effective": 0.12,
        "is_h1n1_risky": 0.08,
        "dr_recc_seasonal_vacc": 0.05,
        "is_seas_vacc_effective": 0.05,
        "is_seas_risky": 0.05,
        "h1n1_worry": 0.04,
        "h1n1_awareness": 0.04,
        "is_health_worker": 0.03,
        "sick_from_h1n1_vacc": 0.03,
        "sick_from_seas_vacc": 0.03,
        "age_bracket": 0.03,
        "contact_avoidance": 0.02,
        "avoid_large_gatherings": 0.02,
        "reduced_outside_home_cont": 0.02,
        "avoid_touch_face": 0.02,
        "chronic_medic_condition": 0.02,
        "qualification": 0.02,
        "census_msa": 0.02,
        "no_of_adults": 0.02,
        "no_of_children": 0.02,
        "antiviral_medication": 0.01,
        "bought_face_mask": 0.01,
        "wash_hands_frequently": 0.01,
        "cont_child_undr_6_mnths": 0.01,
        "race": 0.01,
        "sex": 0.01,
        "income_level": 0.01,
        "marital_status": 0.01,
        "housing_status": 0.01,
        "employment": 0.01,
    }

    # Display the bar chart for feature importance
    st.title("Feature Importance")
    feature_importance_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
    st.bar_chart(feature_importance_df)
    

    
    # Top factors influencing vaccine usage
    st.write("### Top Factors Influencing Vaccine Usage:")
    top_features = sorted(feature_importance, key=feature_importance.get, reverse=True)[:5]
    for i, feature in enumerate(top_features, start=1):
        st.write(f"{i}. {feature.capitalize()} ({feature_importance[feature]*100:.2f}%)")

    # Predictive modeling insights
    st.write("### Predictive Modeling Insights:")
    st.write("- **Accuracy**: The model achieved an accuracy of 82% on the test dataset.")
    st.write("- **Key Factors**: The most important factors influencing vaccine usage include awareness, perceived effectiveness, and perceived risk.")

    # Conclusion
    st.write("### Conclusion:")
    st.write("Understanding the factors that influence vaccine usage is crucial for public health efforts. By targeting outreach and education efforts towards the most influential factors, public health officials can improve vaccination rates and reduce the spread of disease.")

    # Acknowledgements
    st.write("### Acknowledgements:")
    st.write("This project was made possible by the National Flu Survey (NHFS, 2009) dataset and the contributions of public health officials and researchers.")

    
    
