import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
from psycopg2 import sql
import numpy as np

# Database connection function
def create_connection():
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# CRUD Operations
def create_record(table_name, data):
    conn = create_connection()
    if conn:
        try:
            cur = conn.cursor()
            columns = ', '.join(data.keys())
            values = ', '.join(['%s'] * len(data))
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
            cur.execute(query, list(data.values()))
            conn.commit()
            st.success("Record created successfully!")
        except Exception as e:
            st.error(f"Error creating record: {e}")
        finally:
            conn.close()

def read_data(table_name):
    conn = create_connection()
    if conn:
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            st.error(f"Error reading data: {e}")
            return None
        finally:
            conn.close()

def update_record(table_name, record_id, data):
    conn = create_connection()
    if conn:
        try:
            cur = conn.cursor()
            set_values = ', '.join([f"{k} = %s" for k in data.keys()])
            query = f"UPDATE {table_name} SET {set_values} WHERE item_id = %s"
            values = list(data.values()) + [record_id]
            cur.execute(query, values)
            conn.commit()
            st.success("Record updated successfully!")
        except Exception as e:
            st.error(f"Error updating record: {e}")
        finally:
            conn.close()

def delete_record(table_name, record_id):
    conn = create_connection()
    if conn:
        try:
            cur = conn.cursor()
            query = f"DELETE FROM {table_name} WHERE item_id = %s"
            cur.execute(query, (record_id,))
            conn.commit()
            st.success("Record deleted successfully!")
        except Exception as e:
            st.error(f"Error deleting record: {e}")
        finally:
            conn.close()

# Visualization Functions
def create_visualization(df, chart_type, x_col, y_col):
    if chart_type == "Line Chart":
        fig = px.line(df, x=x_col, y=y_col)
    elif chart_type == "Bar Chart":
        fig = px.bar(df, x=x_col, y=y_col)
    elif chart_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col)
    return fig

def display_summary_statistics(df):
    st.subheader("Summary Statistics")
    
    # Numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.write("Numerical Columns:")
        st.write(df[numeric_cols].describe())
    
    # Categorical columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_cols) > 0:
        st.write("Categorical Columns:")
        for col in categorical_cols:
            st.write(f"\n{col} value counts:")
            st.write(df[col].value_counts())

def main():
    st.title("Data Management and Analysis Dashboard")
    
    # Sidebar navigation
    tables = ["items", "table2", "table3", "table4"]
    selected_table = st.sidebar.selectbox("Select Table", tables)
    
    # Read data
    df = read_data(selected_table)
    if df is None:
        st.error("Unable to load data")
        return
    
    # CRUD Operations Section
    st.header("Data Management")
    crud_operation = st.selectbox("Select Operation", ["Create", "Read", "Update", "Delete"])
    
    if crud_operation == "Create":
        st.subheader("Create New Record")
        new_data = {}
        for col in df.columns:
            new_data[col] = st.text_input(f"Enter {col}")
        if st.button("Create Record"):
            create_record(selected_table, new_data)
    
    elif crud_operation == "Read":
        st.subheader("Data Preview")
        st.dataframe(df.head(20))
    
    elif crud_operation == "Update":
        st.subheader("Update Record")
        record_id = st.selectbox("Select Record ID", df['item_id'].unique())
        update_data = {}
        for col in df.columns:
            if col != 'item_id':
                current_value = df[df['item_id'] == record_id][col].iloc[0]
                update_data[col] = st.text_input(f"Enter new {col}", value=str(current_value))
        if st.button("Update Record"):
            update_record(selected_table, record_id, update_data)
    
    elif crud_operation == "Delete":
        st.subheader("Delete Record")
        record_id = st.selectbox("Select Record ID to Delete", df['item_id'].unique())
        if st.button("Delete Record"):
            delete_record(selected_table, record_id)
    
    # Data Visualization Section
    st.header("Data Visualization")
    chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Scatter Plot"])
    
    cols = df.columns.tolist()
    x_col = st.selectbox("Select X-axis column", cols)
    y_col = st.selectbox("Select Y-axis column", cols)
    
    fig = create_visualization(df, chart_type, x_col, y_col)
    st.plotly_chart(fig)
    
    # Filtering Section
    st.header("Data Filtering")
    
    # Numeric filters
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        filter_range = st.slider(f"Filter {col}", min_val, max_val, (min_val, max_val))
        df = df[df[col].between(filter_range[0], filter_range[1])]
    
    # Categorical filters
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        selected_categories = st.multiselect(f"Filter {col}", df[col].unique())
        if selected_categories:
            df = df[df[col].isin(selected_categories)]
    
    # Display filtered data
    st.subheader("Filtered Data")
    st.dataframe(df)
    
    # Summary Statistics
    display_summary_statistics(df)

if __name__ == "__main__":
    main()