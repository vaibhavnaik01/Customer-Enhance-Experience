import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import os


font_scale = 1.2
def load_data():
    file_path = r'C:\Users\Admin\Downloads\No_Show_predicted_labelled.xlsx'
    df = pd.read_excel(file_path)
    df['opened_at_formatted'] = pd.to_datetime(df['opened_at_formatted'], errors='coerce')
    return df

def style_dataframe(df, hide_columns=False):
    if hide_columns:
        columns_to_hide = ['cleaned_description', 'complaints', 'no_show_prediction', 'LABEL', 'complaint_type']
        df = df.drop(columns=columns_to_hide, errors='ignore')
    return df

def style_filtered_dataframe(df):
    columns_to_hide = ['cleaned_description', 'no_show_prediction', 'complaints', 'LABEL']
    df = df.drop(columns=columns_to_hide, errors='ignore')
    return df
def display_visualizations(df, tab1, tab2, tab3, tab4):
      

    with tab1:
        st.markdown("<div class='data-preview'><h3 style='color: #00acce;'>🧾 SN_CUSTOMERSERVICE_CASE - Full Data Preview</h3>", unsafe_allow_html=True)
        styled_df = style_dataframe(df, hide_columns=True)
        st.dataframe(styled_df)
        st.write(f"**Number of records:** {df.shape[0]}")
        st.write(f"**Number of columns currently visible:** {styled_df.shape[1]}")
    
    with tab2:
        st.markdown("<div class='card'><h4 style='color: #00acce;'>📊 Complaint Type Distribution (Bar Chart)</h4>", unsafe_allow_html=True)
        split_df = df.assign(complaint_type=df['complaint_type'].str.split(',')).explode('complaint_type')
        split_df['complaint_type'] = split_df['complaint_type'].str.strip()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.set(font_scale=font_scale)
        sns.countplot(data=split_df, y='complaint_type', hue='complaint_type', order=split_df['complaint_type'].value_counts().index,
                      ax=ax, palette='coolwarm', legend=False)
        ax.set_xlabel("Count")
        ax.set_ylabel("Complaint Type")
        ax.set_title("Complaint Type Distribution")
        st.pyplot(fig)
    
    with tab3:
        st.markdown("<div class='card'><h4 style='color: #00acce;'>📊 Negative Bigrams Distribution (Percentage)</h4>", unsafe_allow_html=True)
        negative_bigrams_df = df[['negative_bigrams']].dropna().copy()
        negative_bigrams_df = negative_bigrams_df['negative_bigrams'].str.split(', ', expand=True).stack().value_counts().reset_index()
        negative_bigrams_df.columns = ['bigram', 'count']

        negative_keywords = ["delay", "damage", "issue", "problem", "failure", "missed", "complaint", "wrong",
                             "error", "fault", "delay", "missing"]
        negative_bigrams_df = negative_bigrams_df[negative_bigrams_df['bigram'].apply(lambda x: any(kw in x for kw in negative_keywords))]

        negative_bigrams_df.loc[:, 'percentage'] = (negative_bigrams_df['count'] / negative_bigrams_df['count'].sum()) * 100
        negative_bigrams_df = negative_bigrams_df.sort_values(by='percentage', ascending=False)
        negative_bigrams_df = negative_bigrams_df.head(10)  

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.set(font_scale=font_scale)
        sns.barplot(data=negative_bigrams_df, y='bigram', x='percentage', hue='bigram', ax=ax, palette='coolwarm', legend=False)
        ax.set_xlabel("Percentage")
        ax.set_ylabel("Negative Bigrams")
        ax.set_title("Top 10 Negative Bigrams by Percentage")
        st.pyplot(fig)
    
    with tab4:
        st.markdown("<div class='card'><h4 style='color: #00acce;'>🌥️ Negative Bigrams Word Cloud</h4>", unsafe_allow_html=True)
        negative_bigrams_series = df['negative_bigrams'].dropna().str.split(', ', expand=True).stack()
        word_freq = negative_bigrams_series.value_counts()
        
        
        unique_bigrams = word_freq[word_freq == 1]
        
        word_dict = unique_bigrams.to_dict()
        fig = generate_wordcloud(word_dict)
        st.pyplot(fig)

def generate_wordcloud(data):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(data)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def main():
    st.set_page_config(page_title="CX Dashboard", layout="wide", page_icon="📊")

    st.markdown("<div class='card' style='background-color:#74c9da;'><h3 style='font-size:45px;color: #000000; text-align: center;'> Welcome to the CX One </h1></div>", unsafe_allow_html=True)
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    with st.sidebar:
        logo_path = r'CITYFIBRE_LOGO.png'
        if os.path.exists(logo_path):
            st.image(logo_path, use_column_width=True)
        else:
            st.warning("Logo not found!")

    
    df = load_data()
    
    min_date = df['opened_at_formatted'].min().date()
    max_date = df['opened_at_formatted'].max().date()

    if pd.isna(min_date) or pd.isna(max_date):
        st.error("No valid dates found in the dataset.")
        return

    st.markdown("<div class='main-content'>", unsafe_allow_html=True)

    
    display_full_data = True
    filtered_df = df.copy()

    
    selected_date_range = st.sidebar.date_input(
        "Select date range:",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    start_date, end_date = selected_date_range

    filtered_df = filtered_df[(filtered_df['opened_at_formatted'] >= pd.to_datetime(start_date)) &
                              (filtered_df['opened_at_formatted'] <= pd.to_datetime(end_date))]

    complaint_types = ["No Show", "Service Activation Issues", "Account Issues", "Awaiting Communication", 
                       "Billing Issues", "Closable Issues", "Connection Issues", "Customer Service Issues", 
                       "Escalated Issues", "Feedback", "Gigabit Issues", "Health and Safety Issues", 
                       "Infrastructure Damage", "Installation Issues", "Maintenance Issue", "Noise Issue", 
                       "Non-Specific Complaints", "Parking Issues", "Post-Work Clean-Up Issues", 
                       "Promotional Offer Issues", "Property Access Issues", "Refund Issues", 
                       "Subscription Issue", "Team Behavior Issues", "Technical Issues", "Work Quality Issues"]

    primary_complaint_type_filter = st.sidebar.multiselect("Select Complaint Type(s):", complaint_types)

    if primary_complaint_type_filter:
        filtered_df = filtered_df[
            (filtered_df['complaints'].isin(primary_complaint_type_filter)) & 
            (filtered_df['no_show_prediction'] == True)
        ]
        display_full_data = False

    tab1, tab2, tab3, tab4 = st.tabs(["📅 Data Preview", "📊 Complaint Type Distribution", "📊 Negative Bigrams Distribution", "🌥️ Negative Bigrams Word Cloud"])

    if display_full_data:
        display_visualizations(df, tab1, tab2, tab3, tab4)
    else:
        with tab1:
            st.markdown("<div class='filtered-data'><h3 style='color: #00acce;'>🔍 Filtered Data</h3>", unsafe_allow_html=True)
            styled_df = style_filtered_dataframe(filtered_df)
            st.dataframe(styled_df)
            st.write(f"**Number of records:** {filtered_df.shape[0]}")
            st.write(f"**Number of columns currently visible:** {styled_df.shape[1]}")

        with tab2:
            st.markdown("<div class='card'><h4 style='color: #00acce;'>📊 Complaint Type Distribution (Bar Chart)</h4>", unsafe_allow_html=True)
            split_df = filtered_df.assign(complaint_type=filtered_df['complaint_type'].str.split(',')).explode('complaint_type')
            split_df['complaint_type'] = split_df['complaint_type'].str.strip()

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.set(font_scale=font_scale)
            sns.countplot(data=split_df, y='complaint_type', hue='complaint_type', order=split_df['complaint_type'].value_counts().index,
                          ax=ax, palette='coolwarm', legend=False)
            ax.set_xlabel("Count")
            ax.set_ylabel("Complaint Type")
            ax.set_title("Complaint Type Distribution")
            st.pyplot(fig)
        
        with tab3:
            st.markdown("<div class='card'><h4 style='color: #00acce;'>📊 Negative Bigrams Distribution (Percentage)</h4>", unsafe_allow_html=True)
            negative_bigrams_df = filtered_df[['negative_bigrams']].dropna().copy()
            negative_bigrams_df = negative_bigrams_df['negative_bigrams'].str.split(', ', expand=True).stack().value_counts().reset_index()
            negative_bigrams_df.columns = ['bigram', 'count']

            negative_keywords = ["delay", "damage", "issue", "problem", "failure", "missed", "complaint", "wrong",
                                 "error", "fault", "delay", "missing"]
            negative_bigrams_df = negative_bigrams_df[negative_bigrams_df['bigram'].apply(lambda x: any(kw in x for kw in negative_keywords))]

            negative_bigrams_df.loc[:, 'percentage'] = (negative_bigrams_df['count'] / negative_bigrams_df['count'].sum()) * 100
            negative_bigrams_df = negative_bigrams_df.sort_values(by='percentage', ascending=False)
            negative_bigrams_df = negative_bigrams_df.head(10)  # Keep only top 10 bigrams

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.set(font_scale=font_scale)
            sns.barplot(data=negative_bigrams_df, y='bigram', x='percentage', hue='bigram', ax=ax, palette='coolwarm', legend=False)
            ax.set_xlabel("Percentage")
            ax.set_ylabel("Negative Bigrams")
            ax.set_title("Top 10 Negative Bigrams by Percentage")
            st.pyplot(fig)
        
        with tab4:
            st.markdown("<div class='card'><h4 style='color: #00acce;'>🌥️ Negative Bigrams Word Cloud</h4>", unsafe_allow_html=True)
            unique_negative_bigrams = filtered_df['negative_bigrams'].dropna().str.split(', ', expand=True).stack().unique()
            unique_bigrams_series = pd.Series(unique_negative_bigrams)
            word_freq = unique_bigrams_series.value_counts()

            word_dict = word_freq.to_dict()
            fig = generate_wordcloud(word_dict)
            st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()