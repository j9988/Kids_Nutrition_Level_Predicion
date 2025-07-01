import pickle
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Children's Nutrition Level Analysis and Prediction",
    page_icon="🌮",  
    layout="wide"   
)

# Load data
@st.cache_data
def load_price_data():
    return pd.read_csv("Dataset/price.csv")

@st.cache_data
def load_kids_data():
    return pd.read_csv("Dataset/kids.csv")

@st.cache_data
def load_model_data():
    return pd.read_csv("Dataset/rf_lor_model_3labels.csv")
    
@st.cache_data
def load_model_and_encoder():
    with open("rf_model_3labels.pkl", "rb") as f:
        model = pickle.load(f)
    with open("ohe_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

# Helper function 
def replace_outliers_with_median(df, column_name):
    """Replaces outliers in a specified column with the median."""
    if column_name not in df.columns or df[column_name].isnull().all():
        return df # Return original if column doesn't exist or is all NaN
    
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_cleaned = df.copy()
    median_value = df[column_name].median()
    
    df_cleaned.loc[(df_cleaned[column_name] < lower_bound) | (df_cleaned[column_name] > upper_bound), column_name] = median_value
    return df_cleaned

def input_with_tooltip(label, tooltip, key, default):
    st.markdown(f"""
        <style>
        .tooltip {{
          position: relative;
          display: inline-block;
        }}

        .tooltip .tooltiptext {{
          visibility: hidden;
          width: 200px;
          background-color: #555;
          color: #fff;
          text-align: center;
          border-radius: 6px;
          padding: 5px;
          position: absolute;
          z-index: 1;
          bottom: 125%;
          left: 50%;
          margin-left: -100px;
          opacity: 0;
          transition: opacity 0.3s;
        }}

        .tooltip:hover .tooltiptext {{
          visibility: visible;
          opacity: 1;
        }}
        </style>

        <div style='font-size:18px; font-weight:500; margin-bottom:4px'>
            {label}
            <span class="tooltip"> 🛈
              <span class="tooltiptext">{tooltip}</span>
            </span>
        </div>
    """, unsafe_allow_html=True)

    return st.number_input(" ", key=key, label_visibility="collapsed", value=default)

## -------------------------------------------------------------------------------

# Sidebar navigation
with st.sidebar:
    page = option_menu(
        "Navigation",  # Menu title
        ["EDA - Price Data", "EDA - Children Data", "Random Forest Prediction"],
        icons=["graph-up", "person-bounding-box", "cpu"],
        menu_icon="list",  # Top icon
        default_index=0  # Default selected page
    )

## -------------------------------------------------------------------------------

if page == "EDA - Price Data":
    st.title("Exploratory Data Analysis (EDA) - Price Data")

    df_price = load_price_data()
    sns.set_palette("pastel")

    st.write("## Price Dataset Sample")
    st.dataframe(df_price.head())
    
    # Ensure 'date' column is datetime for time-series plots
    if 'date' in df_price.columns and not pd.api.types.is_datetime64_any_dtype(df_price['date']):
        try:
            df_price['date'] = pd.to_datetime(df_price['date'])
        except:
             st.warning("Date column could not be converted to datetime for EDA.")
    
    with st.expander("📦 Item Overview", expanded=True):
        st.write("### Distribution of Item Categories")
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.countplot(x='item_category', data=df_price, order=df_price['item_category'].value_counts().index,
                    palette='pastel', ax=ax, hue='item_category', legend=False)
        plt.title('Distribution of Item Categories')
        plt.xlabel('Item Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("### Top 10 Items with Highest Count")
        top_10_items_eda = df_price['item'].value_counts().nlargest(10).reset_index()
        top_10_items_eda.columns = ['item', 'count']
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=top_10_items_eda, x='count', y='item', palette='pastel', 
                    hue='item', legend=False, ax=ax)
        plt.title('Top 10 Items with Highest Count')
        plt.xlabel('Count')
        plt.ylabel('Item')
        plt.tight_layout()
        st.pyplot(fig)
        
    with st.expander("📍 District-Level Insights", expanded=False):
        st.write("### District Ranking by Highest Count")
        district_record_counts = df_price.groupby('district')['premise'].count().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=district_record_counts.index, y=district_record_counts.values, 
                    palette='pastel', hue=district_record_counts.index, legend=False, 
                    ax=ax)
        plt.title('District Ranking by Highest Count')
        plt.xlabel('District')
        plt.ylabel('Number of Price Records')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Top 10 Items within Each District")
        # Create list of unique districts for the dropdown
        districts = df_price['district'].unique()
        selected_district = st.selectbox("Select a District", sorted(districts))
        # Filter top 10 items in selected district
        top_items = (df_price[df_price['district'] == selected_district]
                    .groupby('item').size().sort_values(ascending=False).head(10)
                    .reset_index(name='count'))
        # Display as table
        st.dataframe(top_items)    
    
    with st.expander("💰 Price Analysis", expanded=False):
        st.write("### Price Distribution by Item Category")
        average_prices = df_price.groupby('item_category')['price'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.barplot(x=average_prices.index, y=average_prices.values, palette='pastel', 
                    ax=ax, hue=average_prices.index, legend=False)
        plt.title('Price Distribution by Item Category')
        plt.xlabel('Item Category')
        plt.ylabel('Average Price (RM)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Price Over Time (Selected Items)")
        default_items = ['PISANG BERANGAN', 'BETIK BIASA', 'BERAS CAP FAIZA EMAS (SST5%)',
                        'ROTI SANDWICH GARDENIA ORIGINAL CLASSIC', 'IKAN CENCARU (ANTARA 4 HINGGA 6 EKOR SEKILOGRAM)',
                        'AYAM BERSIH - STANDARD', 'LOBAK MERAH', 'TOMATO', 'KUBIS BUNGA (CAULIFLOWER)', 
                        'SUSU TEPUNG SEGERA EVERYDAY']       
        available_items = df_price['item'].unique()
        default_selection = [item for item in default_items if item in available_items]
        items_to_plot = st.multiselect("Select items to plot:", options=default_items,
                                    default=default_selection[:1]) # Default to first 3 for brevity
        if items_to_plot:
            df_selected_items_time = df_price[df_price['item'].isin(items_to_plot)].copy()
            fig, ax = plt.subplots(figsize=(15, 7))
            for item_name in items_to_plot:
                item_data = df_selected_items_time[df_selected_items_time['item'] == item_name]
                item_data_cleaned = replace_outliers_with_median(item_data, 'price') # Handle outliers per item
                sns.lineplot(x='date', y='price', data=item_data_cleaned, label=item_name, ax=ax, errorbar=None)
            plt.title('Price Over Time')
            plt.xlabel('Date')
            plt.ylabel('Price (RM)')
            plt.xticks(rotation=45)
            plt.legend(title='Item', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)

elif page == "EDA - Children Data":
    st.title("Exploratory Data Analysis (EDA) - Children Data")

    df_kids = load_kids_data()
    sns.set_palette("pastel")

    st.write("## Children Dataset Sample")
    st.dataframe(df_kids.head())
    
    with st.expander("📊 Basic Distributions: Weight, Height & BMI", expanded=True): 
        st.write("### Distribution between Weight and Height")
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.scatterplot(x='BERAT (KG)', y='TINGGI (CM)', data=df_kids, ax=ax, hue='BMI', palette='pastel')
        plt.title('Distribution between Weight and Height')
        plt.xlabel('Weight (kg)')
        plt.ylabel('Height (cm)')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("### Distribution between Age (Months) and BMI")
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.scatterplot(x = 'UMUR (BULAN)', y = 'BMI', data = df_kids, hue = 'BMI', legend = False)
        plt.title('Distribution between Age (Months) and BMI')
        plt.xlabel('Age (Months)')
        plt.ylabel('BMI')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("### Distribution of BMI Classes")
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.countplot(df_kids, x = 'BMI', hue = 'BMI', order = df_kids['BMI'].value_counts().index)
        plt.title('Distribution of Children\'s BMI Classes')
        plt.xlabel('BMI Class')
        plt.ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        
    with st.expander("🧑‍⚕️ BMI by Demographics", expanded=True):
        st.write("### BMI by Gender")
        st.write("#### Distribution between Weight and Height for Boys")
        fig, ax = plt.subplots(figsize=(15, 7))
        p = sns.scatterplot(x = 'BERAT (KG)', y = 'TINGGI (CM)', data = df_kids[df_kids['JANTINA'] == 'LELAKI'], hue = 'BMI')
        plt.title('Distribution between Weight and Height for Boys')
        plt.xlabel('Weight (kg)')
        plt.ylabel('Height (cm)')
        sns.move_legend(p, "lower right")
        st.pyplot(fig)
        
        st.write("#### Distribution between Weight and Height for Girls")
        fig, ax = plt.subplots(figsize=(15, 7))
        p = sns.scatterplot(x = 'BERAT (KG)', y = 'TINGGI (CM)', data = df_kids[df_kids['JANTINA'] == 'PEREMPUAN'], hue = 'BMI')
        plt.title('Distribution between Weight and Height for Girls')
        plt.xlabel('Weight (kg)')
        plt.ylabel('Height (cm)')
        sns.move_legend(p, "lower right")
        st.pyplot(fig)
        
        st.write("### BMI by Household Income")
        df_kids['PENDAPATAN KELUARGA'] = df_kids['PENDAPATAN KELUARGA'].map({
            'Tiada Maklumat': 'Tiada Maklumat',
            'Kumpulan M40 (Pendapatan isi rumah RM 4,360.00 - RM 9,619.00)': 'M40',
            'Kumpulan B40 (Pendapatan isi rumah di bawah RM 4,360.00)': 'B40',
            'Kumpulan T20 (Pendapatan isi rumah RM 9,620.00 ke atas)': 'T20',
            'Miskin (Pendapatan isi rumah kurang RM 980.00)': 'Miskin'})
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.countplot(data=df_kids, x='PENDAPATAN KELUARGA', hue='BMI', order=df_kids['PENDAPATAN KELUARGA'].value_counts().index, ax=ax)
        ax.set_title("BMI by Household Income")
        ax.set_xlabel("Household Income")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
        st.write("### BMI by District")
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.countplot(df_kids, x = 'DAERAH', hue = 'BMI', order = df_kids['DAERAH'].value_counts().index)
        plt.title('BMI by District')
        plt.xlabel('District')
        plt.ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("### Relationship between Age (Months), Gender and BMI")
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.scatterplot(x='UMUR (BULAN)', y='BMI', data=df_kids, hue='JANTINA', palette=['skyblue', 'pink'], ax=ax)
        ax.set_title('Relationship between Age (Months), Gender and BMI')
        ax.set_xlabel('Age (Months)')
        ax.set_ylabel('BMI')
        handles, _ = ax.get_legend_handles_labels()
        new_labels = ['Boys', 'Girls']
        ax.legend(handles=handles[:2], labels=new_labels, title="Gender")  # Only take first 2 if more exist
        st.pyplot(fig)
 
    with st.expander("📍 Location & Taska Insights", expanded=True):        
        st.write("### Distribution of Children's Residential Area")
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.countplot(df_kids, x = 'DAERAH', hue = 'DAERAH', order = df_kids['DAERAH'].value_counts().index)
        plt.title('Distribution of Children\'s Residential Area')
        plt.xlabel('District')
        plt.ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Count of each Type of Taska in each Parliament")
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.countplot(x='PARLIMEN', hue='JENIS TASKA', data=df_kids, order=df_kids['PARLIMEN'].value_counts().index)
        plt.title('Count of each Type of Taska in each Parliament')
        plt.xlabel('Parliament')
        plt.ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig)

        st.write("### Heatmap of Parliament and BMI")
        parliment_bmi_counts = pd.crosstab(df_kids['PARLIMEN'], df_kids['BMI'])
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.heatmap(parliment_bmi_counts, annot=True, cmap="YlGnBu", fmt="d")
        plt.title('Heatmap of Parliament and BMI')
        plt.xlabel('BMI Category')
        plt.ylabel('Parliament')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

elif page == "Random Forest Prediction":
    st.title("Random Forest Prediction (3 Labels)")
    df = load_model_data()
    model, encoder = load_model_and_encoder()
    st.write("Input the following features to predict BMI class:")

    feature_cols = [col for col in df.columns if col not in ["BMI"]]
    categorical_cols = ["STATUS PEMAKANAN", "JENIS TASKA", "TASKA_LOKASI"]  
    
    # Define tooltips
    tooltips = {
        "banana_price": "1 kg, commonly range between RM 4.00 to RM 7.00",
        "papaya_price": "1 kg, commonly range between RM 4.00 to RM 5.00",
        "rice_price": "10 kg, commonly range between RM 25.00 to RM 26.00",
        "bread_price": "1 loaf, commonly range between RM 2.00 to RM 3.00",
        "fish_price": "4-6 fish/kg, commonly range between RM 7.00 to RM 15.00",
        "chicken_price": "1 kg, commonly range between RM 8.00 to RM 10.00",
        "carrot_price": "1 kg, commonly range between RM 4.00 to RM 6.00",
        "tomato_price": "1 kg, commonly range between RM 4.00 to RM 8.00",
        "cauliflower_price": "1 kg, commonly range between RM 7.00 to RM 12.00",
        "milk_price": "0.9 kg, commonly range between RM 16.00 to RM 32.00"
    }
    
    # Collect user input
    user_input = {}
    for col in feature_cols:
        if df[col].dtype == "object":
            user_input[col] = st.selectbox(col, sorted(df[col].dropna().unique()))
        else:
            tooltip = tooltips.get(col, "Enter a numeric value.")
            default_val = float(df[col].median())
            user_input[col] = input_with_tooltip(col, tooltip, key=col, default=default_val)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])

        # One-hot encode categorical columns using the loaded encoder
        input_encoded = encoder.transform(input_df[categorical_cols])
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols))

        # Drop categorical columns and concatenate encoded columns
        input_final = pd.concat([input_df.drop(columns=categorical_cols), input_encoded_df], axis=1)

        # Ensure columns order matches training data
        input_final = input_final.reindex(columns=model.feature_names_in_, fill_value=0)

        prediction = model.predict(input_final)[0]
        st.success(f"Predicted BMI class: **{prediction}**")
