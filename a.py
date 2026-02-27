# --- Streamlit app for user profiling and segmentation ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# --- Streamlit Page Config ---
st.set_page_config(page_title="User Profiling & Segmentation", layout="wide")

# --- Custom CSS for nicer look ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #343a40;}
    .stButton>button {background-color: #4CAF50; color: white;}
    table {width: 100%; border-collapse: collapse;}
    th, td {border: 1px solid #ddd; padding: 8px;}
    th {background-color: #f2f2f2; text-align: left;}
    </style>
""", unsafe_allow_html=True)

# --- App Title & Description ---
st.title("User Profiling & Segmentation App")
st.markdown("""
Upload your user dataset (CSV) to visualize key demographics, 
explore online behavior, and automatically segment your users into *5 well-defined customer groups* using KMeans clustering.
""")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.header("Data Preview")
    st.dataframe(data.head())

    # --- Missing Values Check ---
    with st.expander("Missing Values Report", expanded=False):
        st.write(data.isnull().sum())

    # --- Demographics Distribution ---
    st.header("Demographic Distributions")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Distribution of Key Demographic Variables', fontsize=16, color='darkblue')

    sns.countplot(ax=axes[0, 0], x='Age', data=data, palette='coolwarm')
    axes[0, 0].set_title('Age Distribution', fontsize=14)
    axes[0, 0].tick_params(axis='x', rotation=45)

    sns.countplot(ax=axes[0, 1], x='Gender', data=data, palette='coolwarm')
    axes[0, 1].set_title('Gender Distribution', fontsize=14)

    sns.countplot(ax=axes[1, 0], x='Education Level', data=data, palette='coolwarm')
    axes[1, 0].set_title('Education Level Distribution', fontsize=14)
    axes[1, 0].tick_params(axis='x', rotation=45)

    sns.countplot(ax=axes[1, 1], x='Income Level', data=data, palette='coolwarm')
    axes[1, 1].set_title('Income Level Distribution', fontsize=14)
    axes[1, 1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    # --- Device Usage Distribution ---
    st.header("Device Usage")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Device Usage', data=data, palette='coolwarm')
    plt.title('Device Usage Distribution', fontsize=16)
    st.pyplot(plt.gcf())

    # --- User Online Behavior and Ad Interaction Metrics ---
    st.header("Online Behavior & Ad Interaction")
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('User Online Behavior and Ad Interaction Metrics', fontsize=16, color='darkblue')

    sns.histplot(ax=axes[0, 0], x='Time Spent Online (hrs/weekday)', data=data, bins=20, kde=True, color='skyblue')
    axes[0, 0].set_title('Time Spent Online on Weekdays')

    sns.histplot(ax=axes[0, 1], x='Time Spent Online (hrs/weekend)', data=data, bins=20, kde=True, color='orange')
    axes[0, 1].set_title('Time Spent Online on Weekends')

    sns.histplot(ax=axes[1, 0], x='Likes and Reactions', data=data, bins=20, kde=True, color='green')
    axes[1, 0].set_title('Likes and Reactions')

    sns.histplot(ax=axes[1, 1], x='Click-Through Rates (CTR)', data=data, bins=20, kde=True, color='red')
    axes[1, 1].set_title('Click-Through Rates (CTR)')

    sns.histplot(ax=axes[2, 0], x='Conversion Rates', data=data, bins=20, kde=True, color='purple')
    axes[2, 0].set_title('Conversion Rates')

    sns.histplot(ax=axes[2, 1], x='Ad Interaction Time (sec)', data=data, bins=20, kde=True, color='brown')
    axes[2, 1].set_title('Ad Interaction Time (sec)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig)

    # --- Top User Interests ---
    st.header("Top User Interests")
    interests_list = data['Top Interests'].str.split(', ').sum()
    interests_counter = Counter(interests_list)
    interests_df = pd.DataFrame(interests_counter.items(), columns=['Interest', 'Frequency'])
    interests_df = interests_df.sort_values(by='Frequency', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Frequency',
        y='Interest',
        data=interests_df.head(10),
        palette='coolwarm'
    )
    plt.title('Top 10 User Interests', fontsize=16)
    st.pyplot(plt.gcf())

    # --- Clustering Section ---
    st.header("🔎 User Segmentation with KMeans Clustering (Fixed at 5 Clusters)")

    n_clusters = 5

    features = ['Age', 'Gender', 'Income Level', 'Time Spent Online (hrs/weekday)',
                'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 'Click-Through Rates (CTR)']
    X = data[features]

    numeric_features = ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)',
                        'Likes and Reactions', 'Click-Through Rates (CTR)']
    numeric_transformer = StandardScaler()

    categorical_features = ['Age', 'Gender', 'Income Level']
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('cluster', KMeans(n_clusters=n_clusters, random_state=42))
    ])

    pipeline.fit(X)
    cluster_labels = pipeline.named_steps['cluster'].labels_
    data['Cluster'] = cluster_labels

    # Assign segment names
    segment_name_mapping = {
        0: 'Weekend Warriors',
        1: 'Engaged Professionals',
        2: 'Low-Key Users',
        3: 'Active Explorers',
        4: 'Budget Browsers'
    }
    data['Segment Name'] = data['Cluster'].map(segment_name_mapping)

    st.success("Clustering complete with assigned segment names!")
    st.dataframe(data.head())

    # Download button
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Clustered CSV",
        data=csv,
        file_name='clustered_data_with_segments.csv',
        mime='text/csv'
    )

    # --- Cluster Means + Modes ---
    st.header("Cluster Profiles (Numeric Means + Categorical Modes)")
    cluster_means = data.groupby('Cluster')[numeric_features].mean()
    for feature in categorical_features:
        mode_series = data.groupby('Cluster')[feature].agg(lambda x: x.mode().iloc[0])
        cluster_means[feature] = mode_series
    cluster_means['Segment Name'] = [segment_name_mapping[i] for i in cluster_means.index]
    st.dataframe(cluster_means)

    # --- Radar Chart with Named Segments ---
    st.header("Segment Profiles (Radar Chart)")

    features_to_plot = numeric_features
    labels = np.array(features_to_plot)

    radar_df = cluster_means[features_to_plot].reset_index()

    # Normalize
    radar_df_normalized = radar_df.copy()
    for feature in features_to_plot:
        radar_df_normalized[feature] = (
            radar_df[feature] - radar_df[feature].min()
        ) / (radar_df[feature].max() - radar_df[feature].min())

    radar_df_normalized = pd.concat([radar_df_normalized, radar_df_normalized.iloc[[0]]], ignore_index=True)

    fig = go.Figure()
    for i in range(n_clusters):
        values = radar_df_normalized.iloc[i][features_to_plot]
        fig.add_trace(go.Scatterpolar(
            r=values.tolist() + [values.iloc[0]],
            theta=labels.tolist() + [labels[0]],
            fill='toself',
            name=segment_name_mapping[i],
            hoverinfo='text',
            text=[f"{label}: {value:.2f}" for label, value in zip(features_to_plot, values)] +
                 [f"{labels[0]}: {values.iloc[0]:.2f}"]
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='User Segments Profile (Radar Chart)'
    )
    st.plotly_chart(fig)

    # --- BOTH tables side by side below ---
    st.header("🗂️ Segment Naming Justification")
    st.markdown("""
These segment names were carefully chosen based on data patterns, making them intuitive for marketing teams to use.

| **Segment Name**         | **Why This Name? (Data Justification)** | **Example User Types** |
|---------------------------|-----------------------------------------|-------------------------|
| **Weekend Warriors**      | High weekend online time vs. weekdays. Age 25–34, mid-to-high income. Suggests busy weekdays with leisure browsing on weekends. | Young working professionals, college students |
| **Engaged Professionals** | Low weekday usage, very high weekend usage, highest CTR. Age 35–44, higher income. Implies career-focused people who browse purposefully and engage with ads when they have time. | Mid-career professionals, managers, business owners |
| **Low-Key Users**         | Moderate online time with slightly lower hours on weekends, mid-range income, *very high* Likes and Reactions. Balanced but socially active users who interact a lot despite lower total time online. | Young adults with steady jobs, casual social-media users |
| **Active Explorers**      | High usage on both weekdays and weekends, highest Likes and Reactions, high income (100k+), age 25–34. Indicates people who love discovering new content and engaging widely across platforms. | Affluent millennials, content creators |
| **Budget Browsers**       | Older users (45–54) with lower income (0–20k), less time online, lower engagement. Budget-conscious and value-oriented browsing behavior. | Retirees, lower-income older adults |
    """, unsafe_allow_html=True)

    st.header("🎯 Marketing Strategy & Creative Ideas")
    st.markdown("""
These user segments allow marketers to design tailored strategies that resonate with real customer behavior.

| **Segment Name**         | **How to Leverage This Segment** | **Creative Marketing & Strategy Ideas** |
|---------------------------|----------------------------------|----------------------------------------|
| **Weekend Warriors**      | Capture their attention when they’re most active—on weekends. They want to relax, explore, and shop online during free time. | 🎯 Weekend flash sales <br> 🎬 Entertaining content drops (videos, blogs) timed for Fridays <br> 📣 Push notifications for weekend-only deals |
| **Engaged Professionals** | Appeal to busy, career-driven users with purposeful and premium experiences. They have high ad engagement but limited time. | 💼 Targeted LinkedIn ads <br> 🗓️ Early-morning/evening email offers <br> 🏷️ Premium subscriptions or concierge services |
| **Low-Key Users**         | Consistent, socially active but moderate overall time. They appreciate value and relatable content. | 🤝 Community-building campaigns <br> 👍 Social-proof-heavy ads with testimonials <br> 🏠 Home essentials, daily-use product bundles |
| **Active Explorers**      | Highly active, curious, affluent users who love trying new things. Ideal for launching and spreading new ideas. | 🚀 Early-access programs <br> 🌟 Influencer/creator collaborations <br> 📲 Gamified referral systems to encourage viral sharing |
| **Budget Browsers**       | Value-conscious, older users with lower income. They want clear, trustworthy, affordable options. | 💰 Discounts, loyalty rewards <br> 🛒 Easy, accessible UX with large fonts <br> 📢 Ads that emphasize savings, reliability, and simplicity |
    """, unsafe_allow_html=True)

else:
    st.info("Upload a CSV file to get started!")
