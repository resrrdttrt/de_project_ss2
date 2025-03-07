import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import numpy as np
from contextlib import contextmanager
import time

# Set page configuration
st.set_page_config(
    page_title="Textile Industry Management System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .right-align {
        text-align: right;
    }
    .center-align {
        text-align: center;
    }
    .st-emotion-cache-16txtl3 h1 {
        text-align: center;
        color: #1E88E5;
    }
    div[data-testid="stSidebarUserContent"] hr {
        margin-top: 30px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Database connection functionality
@contextmanager
def connect_to_db():
    """Create a connection to the PostgreSQL database."""
    # Replace with your database credentials
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="airflow",
        password="airflow",
        port="5433"
    )
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize database if not already set up."""
    with connect_to_db() as conn:
        cursor = conn.cursor()
        # Check if table exists, if not create it
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'items'
            );
        """)
        if not cursor.fetchone()[0]:
            cursor.execute("""
                CREATE TABLE items (
                    item_id VARCHAR(50) PRIMARY KEY,
                    item_fit_type VARCHAR(50),
                    item_fabric VARCHAR(50),
                    item_fabric_amount FLOAT,
                    item_mrp FLOAT
                );
                
                -- Insert some sample data
                INSERT INTO items VALUES
                    ('TSH15', 'Oversize', 'Linen', 9.3, 249.8092),
                    ('TSH16', 'Slim', 'Cotton', 8.5, 199.99),
                    ('TSH17', 'Regular', 'Polyester', 7.8, 149.50),
                    ('TSH18', 'Slim', 'Silk', 6.5, 299.99),
                    ('TSH19', 'Oversize', 'Cotton', 10.2, 219.75);
            """)
            conn.commit()

def fetch_items(page=0, items_per_page=20, filters=None):
    """Fetch items from database with pagination and filtering."""
    offset = page * items_per_page
    
    # Base query
    query = """
        SELECT 
            item_id, item_fit_type, item_fabric, item_fabric_amount, item_mrp
        FROM items
    """
    
    # Add filters if provided
    where_clauses = []
    params = []
    
    if filters:
        if filters.get('fit_types'):
            placeholders = ', '.join(['%s'] * len(filters['fit_types']))
            where_clauses.append(f"item_fit_type IN ({placeholders})")
            params.extend(filters['fit_types'])
            
        if filters.get('fabrics'):
            placeholders = ', '.join(['%s'] * len(filters['fabrics']))
            where_clauses.append(f"item_fabric IN ({placeholders})")
            params.extend(filters['fabrics'])
            
        if filters.get('min_fabric_amount') is not None:
            where_clauses.append("item_fabric_amount >= %s")
            params.append(filters['min_fabric_amount'])
            
        if filters.get('max_fabric_amount') is not None:
            where_clauses.append("item_fabric_amount <= %s")
            params.append(filters['max_fabric_amount'])
            
        if filters.get('min_mrp') is not None:
            where_clauses.append("item_mrp >= %s")
            params.append(filters['min_mrp'])
            
        if filters.get('max_mrp') is not None:
            where_clauses.append("item_mrp <= %s")
            params.append(filters['max_mrp'])
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    # Add pagination
    query += f" LIMIT {items_per_page} OFFSET {offset}"
    
    with connect_to_db() as conn:
        # Get items
        df = pd.read_sql(query, conn, params=params)
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM items"
        if where_clauses:
            count_query += " WHERE " + " AND ".join(where_clauses)
        total_items = pd.read_sql(count_query, conn, params=params).iloc[0, 0]
        
    return df, total_items

def get_distinct_values(column):
    """Fetch distinct values for a column."""
    with connect_to_db() as conn:
        query = f"SELECT DISTINCT {column} FROM items ORDER BY {column}"
        return pd.read_sql(query, conn)[column].tolist()

def get_value_ranges():
    """Get min and max values for numeric columns."""
    with connect_to_db() as conn:
        query = """
            SELECT 
                MIN(item_fabric_amount) as min_fabric, 
                MAX(item_fabric_amount) as max_fabric,
                MIN(item_mrp) as min_mrp,
                MAX(item_mrp) as max_mrp
            FROM items
        """
        return pd.read_sql(query, conn).iloc[0].to_dict()

def insert_item(item_data):
    """Insert a new item into the database."""
    with connect_to_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO items (item_id, item_fit_type, item_fabric, item_fabric_amount, item_mrp)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            item_data['item_id'],
            item_data['item_fit_type'],
            item_data['item_fabric'],
            item_data['item_fabric_amount'],
            item_data['item_mrp']
        ))
        conn.commit()

def update_item(item_data):
    """Update an existing item in the database."""
    with connect_to_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE items
            SET item_fit_type = %s, item_fabric = %s, item_fabric_amount = %s, item_mrp = %s
            WHERE item_id = %s
        """, (
            item_data['item_fit_type'],
            item_data['item_fabric'],
            item_data['item_fabric_amount'],
            item_data['item_mrp'],
            item_data['item_id']
        ))
        conn.commit()

def delete_item(item_id):
    """Delete an item from the database."""
    with connect_to_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM items WHERE item_id = %s", (item_id,))
        conn.commit()

def get_all_items_for_viz():
    """Fetch all items for visualization and statistics."""
    with connect_to_db() as conn:
        return pd.read_sql("SELECT * FROM items", conn)

def get_item_by_id(item_id):
    """Fetch a specific item by ID."""
    with connect_to_db() as conn:
        query = "SELECT * FROM items WHERE item_id = %s"
        return pd.read_sql(query, conn, params=(item_id,))

def check_item_exists(item_id):
    """Check if an item with the given ID already exists."""
    with connect_to_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM items WHERE item_id = %s", (item_id,))
        return cursor.fetchone() is not None

# Initialize database on app startup
init_database()

# ---- MAIN APP ----
def main():
    # Create the option menu in the sidebar
    with st.sidebar:
        st.title("Textile Industry Management")
        
        # Main menu options
        main_menu = option_menu(
            menu_title=None,
            options=["Items", "Stores", "Transaction", "Realtime", "Predict" "About"],
            icons=["table", "shop", "cash-coin", "broadcast", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

        # Sub-options as a dropdown for "Items", "Stores", and "Transaction"
        if main_menu == "Items":
            st.markdown("#### Items")
            selected_submenu = st.selectbox(
                "Select an option", 
                ["Data", "Visualization", "Statistics"], 
                index=0
            )
        elif main_menu == "Stores":
            st.markdown("#### Stores")
            selected_submenu = st.selectbox(
                "Select an option", 
                ["Data", "Visualization", "Statistics"], 
                index=0
            )
        elif main_menu == "Transaction":
            st.markdown("#### Transaction")
            selected_submenu = st.selectbox(
                "Select an option", 
                ["Data", "Visualization", "Statistics"], 
                index=0
            )
        elif main_menu == "Realtime":
            st.markdown("#### Realtime")
            selected_submenu = st.selectbox(
                "Select an option", 
                ["Data", "Visualization", "Statistics"], 
                index=0
            )
        st.markdown("---")
        st.caption("© 2025 Textile Industry Management System")

    # Display the selected section based on the main menu and dropdown selection
    if main_menu == "Items":
        if selected_submenu == "Data":
            display_items_page()
        elif selected_submenu == "Visualization":
            display_visualization_page()
        elif selected_submenu == "Statistics":
            display_statistics_page()

    elif main_menu == "Stores":
        if selected_submenu == "Data":
            display_stores_data_page()
        elif selected_submenu == "Visualization":
            display_stores_visualization_page()
        elif selected_submenu == "Statistics":
            display_stores_statistics_page()

    elif main_menu == "Transaction":
        if selected_submenu == "Data":
            display_transaction_data_page()
        elif selected_submenu == "Visualization":
            display_transaction_visualization_page()
        elif selected_submenu == "Statistics":
            display_transaction_statistics_page()

    elif main_menu == "Realtime":
        if selected_submenu == "Data":
            display_realtime_data_page()
        elif selected_submenu == "Visualization":
            display_realtime_visualization_page()
        elif selected_submenu == "Statistics":
            display_realtime_statistics_page()

    elif main_menu == "About":
        display_about_page()

    elif main_menu == "Predict":
        display_predict_page()


# ---- ITEMS PAGE ----
def display_items_page():
    st.title("Items Management")
    
    # Initialize session state variables if they don't exist
    if 'page_num' not in st.session_state:
        st.session_state.page_num = 0
    if 'items_per_page' not in st.session_state:
        st.session_state.items_per_page = 20
    if 'editing' not in st.session_state:
        st.session_state.editing = False
    if 'edit_item' not in st.session_state:
        st.session_state.edit_item = {}
    if 'confirming_delete' not in st.session_state:
        st.session_state.confirming_delete = None
    if 'adding_new' not in st.session_state:
        st.session_state.adding_new = False
    if 'filters' not in st.session_state:
        st.session_state.filters = {}
    
    # Create layout with columns
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Filters")
        st.markdown("---")
        
        # Get distinct values and ranges for filtering
        fit_types = get_distinct_values("item_fit_type")
        fabrics = get_distinct_values("item_fabric")
        ranges = get_value_ranges()
        
        # Filtering widgets
        selected_fit_types = st.multiselect(
            "Fit Type", 
            options=fit_types,
            default=st.session_state.filters.get('fit_types', [])
        )
        
        selected_fabrics = st.multiselect(
            "Fabric", 
            options=fabrics,
            default=st.session_state.filters.get('fabrics', [])
        )
        
        mrp_range = st.slider(
            "MRP Range",
            min_value=float(ranges['min_mrp']),
            max_value=float(ranges['max_mrp']),
            value=(
                st.session_state.filters.get('min_mrp', float(ranges['min_mrp'])),
                st.session_state.filters.get('max_mrp', float(ranges['max_mrp']))
            ),
            step=0.01
        )
        
        # Update filters in session state
        filters = {}
        if selected_fit_types:
            filters['fit_types'] = selected_fit_types
        if selected_fabrics:
            filters['fabrics'] = selected_fabrics
        
        filters['min_mrp'] = mrp_range[0]
        filters['max_mrp'] = mrp_range[1]
        
        # Apply filters button
        if st.button("Apply Filters"):
            st.session_state.filters = filters
            st.session_state.page_num = 0  # Reset to first page
            st.rerun()
        
        # Reset filters button
        if st.button("Reset Filters"):
            st.session_state.filters = {}
            st.session_state.page_num = 0  # Reset to first page
            st.rerun()
    
    with col1:
        # Add new item button
        if not st.session_state.adding_new and not st.session_state.editing:
            if st.button("Add New Item"):
                st.session_state.adding_new = True
                st.rerun()
        
        # Form for adding new item
        if st.session_state.adding_new:
            st.markdown("### Add New Item")
            st.markdown("---")
            
            with st.form(key="add_item_form"):
                new_item_id = st.text_input("Item ID", key="new_id")
                new_fit_type = st.selectbox("Fit Type", options=fit_types, key="new_fit")
                new_fabric = st.text_input("Fabric", key="new_fabric")
                new_fabric_amount = st.number_input("Fabric Amount", min_value=0.1, step=0.1, key="new_amount")
                new_mrp = st.number_input("MRP", min_value=0.01, step=0.01, key="new_mrp")
                
                col1, col2 = st.columns(2)
                submitted = col1.form_submit_button("Save")
                if col2.form_submit_button("Cancel"):
                    st.session_state.adding_new = False
                    st.rerun()
                
                if submitted:
                    if not new_item_id:
                        st.error("Item ID is required")
                    elif check_item_exists(new_item_id):
                        st.error(f"Item with ID {new_item_id} already exists")
                    else:
                        item_data = {
                            'item_id': new_item_id,
                            'item_fit_type': new_fit_type,
                            'item_fabric': new_fabric,
                            'item_fabric_amount': new_fabric_amount,
                            'item_mrp': new_mrp
                        }
                        insert_item(item_data)
                        st.session_state.adding_new = False
                        st.success(f"Item {new_item_id} added successfully!")
                        time.sleep(1)
                        st.rerun()
        
        # Form for editing item
        if st.session_state.editing:
            st.markdown(f"### Edit Item: {st.session_state.edit_item['item_id']}")
            st.markdown("---")
            
            with st.form(key="edit_item_form"):
                edit_item = st.session_state.edit_item
                edit_fit_type = st.selectbox("Fit Type", options=fit_types, index=fit_types.index(edit_item['item_fit_type']) if edit_item['item_fit_type'] in fit_types else 0)
                edit_fabric = st.text_input("Fabric", value=edit_item['item_fabric'])
                edit_fabric_amount = st.number_input("Fabric Amount", value=float(edit_item['item_fabric_amount']), min_value=0.1, step=0.1)
                edit_mrp = st.number_input("MRP", value=float(edit_item['item_mrp']), min_value=0.01, step=0.01)
                
                col1, col2 = st.columns(2)
                submitted = col1.form_submit_button("Update")
                if col2.form_submit_button("Cancel"):
                    st.session_state.editing = False
                    st.rerun()
                
                if submitted:
                    item_data = {
                        'item_id': edit_item['item_id'],
                        'item_fit_type': edit_fit_type,
                        'item_fabric': edit_fabric,
                        'item_fabric_amount': edit_fabric_amount,
                        'item_mrp': edit_mrp
                    }
                    update_item(item_data)
                    st.session_state.editing = False
                    st.success(f"Item {edit_item['item_id']} updated successfully!")
                    time.sleep(1)
                    st.rerun()
        
        # Delete confirmation
        if st.session_state.confirming_delete:
            st.markdown(f"### Confirm Delete: {st.session_state.confirming_delete}")
            st.markdown("---")
            st.warning("Are you sure you want to delete this item? This action cannot be undone.")
            
            col1, col2 = st.columns(2)
            if col1.button("Yes, Delete"):
                delete_item(st.session_state.confirming_delete)
                st.session_state.confirming_delete = None
                st.success("Item deleted successfully!")
                time.sleep(1)
                st.rerun()
            if col2.button("Cancel"):
                st.session_state.confirming_delete = None
                st.rerun()
        
        # Only show the data table if not adding, editing, or confirming delete
        if not st.session_state.adding_new and not st.session_state.editing and not st.session_state.confirming_delete:
            # Fetch items with pagination and filtering
            items_df, total_items = fetch_items(
                page=st.session_state.page_num,
                items_per_page=st.session_state.items_per_page,
                filters=st.session_state.filters
            )
            
            # Display total item count and current range
            total_pages = (total_items - 1) // st.session_state.items_per_page + 1
            start_item = st.session_state.page_num * st.session_state.items_per_page + 1
            end_item = min(start_item + st.session_state.items_per_page - 1, total_items)
            
            st.markdown(f"Showing {start_item} to {end_item} of {total_items} items")
            
            # Display the items table
            if not items_df.empty:
                # Create an interactive table with Streamlit
                st.dataframe(
                    items_df,
                    column_config={
                        "item_id": "Item ID",
                        "item_fit_type": "Fit Type",
                        "item_fabric": "Fabric",
                        "item_fabric_amount": st.column_config.NumberColumn("Fabric Amount", format="%.1f"),
                        "item_mrp": st.column_config.NumberColumn("MRP (₹)", format="%.2f"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Add pagination controls
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.session_state.page_num > 0:
                        if st.button("← Previous"):
                            st.session_state.page_num -= 1
                            st.rerun()
                with col2:
                    st.markdown(f"<div class='center-align'>Page {st.session_state.page_num + 1} of {total_pages}</div>", unsafe_allow_html=True)
                with col3:
                    if st.session_state.page_num < total_pages - 1:
                        if st.button("Next →"):
                            st.session_state.page_num += 1
                            st.rerun()
                
                # Add action buttons for each row
                st.markdown("### Actions")
                selected_item = st.selectbox("Select an item to edit or delete:", options=items_df['item_id'].tolist())
                
                col1, col2 = st.columns(2)
                if col1.button("Edit Selected Item"):
                    item_data = get_item_by_id(selected_item).iloc[0].to_dict()
                    st.session_state.edit_item = item_data
                    st.session_state.editing = True
                    st.rerun()
                
                if col2.button("Delete Selected Item"):
                    st.session_state.confirming_delete = selected_item
                    st.rerun()
            else:
                st.info("No items found matching your criteria.")

# ---- VISUALIZATION PAGE ----
def display_visualization_page():
    st.title("Data Visualization")
    
    # Fetch all items for visualization
    items_df = get_all_items_for_viz()
    
    if items_df.empty:
        st.info("No items found in the database.")
        return
    
    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization Type",
        options=["Bar Chart", "Pie Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot"]
    )
    
    # Column selection based on visualization type
    if viz_type in ["Bar Chart", "Pie Chart"]:
        st.markdown("### Chart Configuration")
        col1, col2 = st.columns(2)
        with col1:
            category_col = st.selectbox(
                "Select Category Column",
                options=["item_fit_type", "item_fabric"],
                format_func=lambda x: x.replace("item_", "").replace("_", " ").title()
            )
        with col2:
            value_col = st.selectbox(
                "Select Value Column",
                options=["item_fabric_amount", "item_mrp", "count"],
                format_func=lambda x: "Count" if x == "count" else x.replace("item_", "").replace("_", " ").title()
            )
        
        # Prepare data for visualization
        if value_col == "count":
            chart_data = items_df.groupby(category_col).size().reset_index(name='count')
            y_col = "count"
            y_title = "Count"
        else:
            chart_data = items_df.groupby(category_col)[value_col].sum().reset_index()
            y_col = value_col
            y_title = value_col.replace("item_", "").replace("_", " ").title()
        
        # Create visualization
        if viz_type == "Bar Chart":
            fig = px.bar(
                chart_data,
                x=category_col,
                y=y_col,
                color=category_col,
                labels={
                    category_col: category_col.replace("item_", "").replace("_", " ").title(),
                    y_col: y_title
                },
                title=f"{y_title} by {category_col.replace('item_', '').replace('_', ' ').title()}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Pie Chart":
            fig = px.pie(
                chart_data,
                values=y_col,
                names=category_col,
                title=f"{y_title} Distribution by {category_col.replace('item_', '').replace('_', ' ').title()}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Line Chart":
        st.markdown("### Chart Configuration")
        
        # For line chart, we'll create a trend over item_id (sorted)
        sorted_df = items_df.sort_values('item_id')
        
        y_col = st.selectbox(
            "Select Value Column",
            options=["item_fabric_amount", "item_mrp"],
            format_func=lambda x: x.replace("item_", "").replace("_", " ").title()
        )
        
        fig = px.line(
            sorted_df,
            x='item_id',
            y=y_col,
            markers=True,
            labels={
                'item_id': 'Item ID',
                y_col: y_col.replace("item_", "").replace("_", " ").title()
            },
            title=f"Trend of {y_col.replace('item_', '').replace('_', ' ').title()} by Item ID"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot":
        st.markdown("### Chart Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox(
                "Select X-Axis Column",
                options=["item_fabric_amount", "item_mrp"],
                format_func=lambda x: x.replace("item_", "").replace("_", " ").title()
            )
        with col2:
            y_col = st.selectbox(
                "Select Y-Axis Column",
                options=["item_fabric_amount", "item_mrp"],
                index=1 if x_col == "item_fabric_amount" else 0,
                format_func=lambda x: x.replace("item_", "").replace("_", " ").title()
            )
        with col3:
            color_col = st.selectbox(
                "Color By",
                options=[None, "item_fit_type", "item_fabric"],
                format_func=lambda x: "None" if x is None else x.replace("item_", "").replace("_", " ").title()
            )
        
        fig = px.scatter(
            items_df,
            x=x_col,
            y=y_col,
            color=color_col,
            hover_name="item_id",
            labels={
                x_col: x_col.replace("item_", "").replace("_", " ").title(),
                y_col: y_col.replace("item_", "").replace("_", " ").title(),
                color_col: color_col.replace("item_", "").replace("_", " ").title() if color_col else None
            },
            title=f"Relationship between {x_col.replace('item_', '').replace('_', ' ').title()} and {y_col.replace('item_', '').replace('_', ' ').title()}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram":
        st.markdown("### Chart Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            hist_col = st.selectbox(
                "Select Column for Histogram",
                options=["item_fabric_amount", "item_mrp"],
                format_func=lambda x: x.replace("item_", "").replace("_", " ").title()
            )
        with col2:
            bin_count = st.slider("Number of Bins", min_value=5, max_value=20, value=10)
        
        fig = px.histogram(
            items_df,
            x=hist_col,
            nbins=bin_count,
            labels={
                hist_col: hist_col.replace("item_", "").replace("_", " ").title()
            },
            title=f"Distribution of {hist_col.replace('item_', '').replace('_', ' ').title()}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        st.markdown("### Chart Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            y_col = st.selectbox(
                "Select Value Column",
                options=["item_fabric_amount", "item_mrp"],
                format_func=lambda x: x.replace("item_", "").replace("_", " ").title()
            )
        with col2:
            category_col = st.selectbox(
                "Group By (Optional)",
                options=[None, "item_fit_type", "item_fabric"],
                format_func=lambda x: "None" if x is None else x.replace("item_", "").replace("_", " ").title()
            )
        
        if category_col:
            fig = px.box(
                items_df,
                y=y_col,
                x=category_col,
                labels={
                    y_col: y_col.replace("item_", "").replace("_", " ").title(),
                    category_col: category_col.replace("item_", "").replace("_", " ").title()
                },
                title=f"Distribution of {y_col.replace('item_', '').replace('_', ' ').title()} by {category_col.replace('item_', '').replace('_', ' ').title()}"
            )
        else:
            fig = px.box(
                items_df,
                y=y_col,
                labels={
                    y_col: y_col.replace("item_", "").replace("_", " ").title()
                },
                title=f"Distribution of {y_col.replace('item_', '').replace('_', ' ').title()}"
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Add visualization description
    st.markdown("### Insights")
    if viz_type == "Bar Chart":
        st.write("""
        The bar chart shows the total values for each category. Taller bars indicate higher values.
        This visualization is useful for comparing categories and identifying which categories have the highest values.
        """)
    elif viz_type == "Pie Chart":
        st.write("""
        The pie chart shows how the selected value is distributed across different categories.
        Larger slices represent a greater proportion of the total.
        This is useful for understanding the relative contribution of each category.
        """)
    elif viz_type == "Line Chart":
        st.write("""
        The line chart shows how values change across items (sorted by Item ID).
        This helps identify trends, patterns, or anomalies in the data across different items.
        """)
    elif viz_type == "Scatter Plot":
        st.write("""
        The scatter plot shows the relationship between two numeric variables.
        Each point represents an item, and clustering patterns can reveal correlations between the variables.
        A positive correlation means both values tend to increase together, while a negative correlation means one increases as the other decreases.
        """)
    elif viz_type == "Histogram":
        st.write("""
        The histogram shows the distribution of values for the selected numeric column.
        Taller bars indicate that more items fall within that range of values.
        This helps identify common values, outliers, and the overall shape of the distribution.
        """)
    elif viz_type == "Box Plot":
        st.write("""
        The box plot shows the distribution of values with key statistics:
        - The box represents the middle 50% of values
        - The line in the box is the median
        - The whiskers extend to the minimum and maximum values (excluding outliers)
        - Points beyond the whiskers are potential outliers
        This visualization helps understand the central tendency, spread, and skewness of the data.
        """)

# ---- STATISTICS PAGE ----
def display_statistics_page():
    st.title("Summary Statistics")
    
    # Fetch all items for statistics
    items_df = get_all_items_for_viz()
    
    if items_df.empty:
        st.info("No items found in the database.")
        return
    
    st.markdown("### Overall Statistics")
    
    # Create tabs for different types of statistics
    tab1, tab2, tab3 = st.tabs(["Numeric Columns", "Categorical Columns", "Correlation"])
    
    with tab1:
        # Numeric columns statistics
        numeric_cols = ["item_fabric_amount", "item_mrp"]
        numeric_stats = items_df[numeric_cols].describe().T
        
        # Format the statistics table
        st.dataframe(
            numeric_stats,
            column_config={
                "count": st.column_config.NumberColumn("Count", format="%d"),
                "mean": st.column_config.NumberColumn("Mean", format="%.2f"),
                "std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                "min": st.column_config.NumberColumn("Min", format="%.2f"),
                "25%": st.column_config.NumberColumn("25th Percentile", format="%.2f"),
                "50%": st.column_config.NumberColumn("Median", format="%.2f"),
                "75%": st.column_config.NumberColumn("75th Percentile", format="%.2f"),
                "max": st.column_config.NumberColumn("Max", format="%.2f"),
            },
            use_container_width=True
        )
        
        # Add explanation of statistics
        st.markdown("""
        **Understanding the statistics:**
        - **Count**: Number of non-null values
        - **Mean**: Average value
        - **Std Dev**: Standard deviation (measure of dispersion)
        - **Min**: Minimum value
        - **25th Percentile**: 25% of values are below this
        - **Median**: Middle value (50th percentile)
        - **75th Percentile**: 75% of values are below this
        - **Max**: Maximum value
        """)
        
        # Additional insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Fabric Amount Insights")
            total_fabric = items_df["item_fabric_amount"].sum()
            avg_fabric = items_df["item_fabric_amount"].mean()
            
            st.metric("Total Fabric Used", f"{total_fabric:.2f} units")
            st.metric("Average Fabric per Item", f"{avg_fabric:.2f} units")
            
            # Most and least fabric-intensive items
            max_fabric_item = items_df.loc[items_df["item_fabric_amount"].idxmax()]
            min_fabric_item = items_df.loc[items_df["item_fabric_amount"].idxmin()]
            
            st.write(f"**Most fabric-intensive item:** {max_fabric_item['item_id']} ({max_fabric_item['item_fabric_amount']:.2f} units)")
            st.write(f"**Least fabric-intensive item:** {min_fabric_item['item_id']} ({min_fabric_item['item_fabric_amount']:.2f} units)")
        
        with col2:
            st.markdown("#### MRP Insights")
            total_mrp = items_df["item_mrp"].sum()
            avg_mrp = items_df["item_mrp"].mean()
            
            st.metric("Total MRP Value", f"₹{total_mrp:.2f}")
            st.metric("Average MRP per Item", f"₹{avg_mrp:.2f}")
            
            # Most and least expensive items
            max_mrp_item = items_df.loc[items_df["item_mrp"].idxmax()]
            min_mrp_item = items_df.loc[items_df["item_mrp"].idxmin()]
            
            st.write(f"**Most expensive item:** {max_mrp_item['item_id']} (₹{max_mrp_item['item_mrp']:.2f})")
            st.write(f"**Least expensive item:** {min_mrp_item['item_id']} (₹{min_mrp_item['item_mrp']:.2f})")
    
    with tab2:
        # Categorical columns statistics
        cat_cols = ["item_fit_type", "item_fabric"]
        
        for col in cat_cols:
            st.markdown(f"#### {col.replace('item_', '').replace('_', ' ').title()} Distribution")
            
            # Count values
            value_counts = items_df[col].value_counts().reset_index()
            value_counts.columns = [col.replace('item_', '').replace('_', ' ').title(), "Count"]
            
            # Calculate percentages
            value_counts["Percentage"] = (value_counts["Count"] / value_counts["Count"].sum() * 100).round(2)
            
            # Display as table
            st.dataframe(
                value_counts,
                column_config={
                    col.replace('item_', '').replace('_', ' ').title(): st.column_config.TextColumn(col.replace('item_', '').replace('_', ' ').title()),
                    "Count": st.column_config.NumberColumn("Count", format="%d"),
                    "Percentage": st.column_config.NumberColumn("Percentage", format="%.2f%%"),
                },
                use_container_width=True
            )
            
            # Create a pie chart
            fig = px.pie(
                value_counts, 
                values="Count", 
                names=col.replace('item_', '').replace('_', ' ').title(),
                title=f"{col.replace('item_', '').replace('_', ' ').title()} Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Correlation analysis
        st.markdown("#### Correlation Between Numeric Variables")
        
        # Calculate correlation matrix
        corr_matrix = items_df[numeric_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix",
            labels=dict(color="Correlation"),
            x=[col.replace('item_', '').replace('_', ' ').title() for col in numeric_cols],
            y=[col.replace('item_', '').replace('_', ' ').title() for col in numeric_cols]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation of correlation
        st.markdown("""
        **Understanding correlation:**
        - Values range from -1 to 1
        - 1: Perfect positive correlation (as one variable increases, the other increases proportionally)
        - 0: No correlation (variables are independent)
        - -1: Perfect negative correlation (as one variable increases, the other decreases proportionally)
        """)
        
        # Fabric amount vs MRP analysis
        st.markdown("#### Fabric Amount vs MRP Analysis")
        
        # Create scatter plot
        fig = px.scatter(
            items_df,
            x="item_fabric_amount",
            y="item_mrp",
            hover_data=["item_id", "item_fit_type", "item_fabric"],
            trendline="ols",
            labels={
                "item_fabric_amount": "Fabric Amount",
                "item_mrp": "MRP (₹)"
            },
            title="Relationship between Fabric Amount and MRP"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate linear regression statistics
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(items_df["item_fabric_amount"], items_df["item_mrp"])
        
        # Display regression statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R-squared", f"{r_value**2:.4f}")
            st.write("**Interpretation:** Higher values (closer to 1) indicate a stronger linear relationship")
            
            if p_value < 0.05:
                st.success(f"The relationship is statistically significant (p-value: {p_value:.4f})")
            else:
                st.warning(f"The relationship is not statistically significant (p-value: {p_value:.4f})")
                
        with col2:
            st.metric("Slope", f"{slope:.4f}")
            st.write(f"**Interpretation:** For each additional unit of fabric, MRP increases by ₹{slope:.2f} on average")
            
            st.metric("Intercept", f"{intercept:.4f}")
            st.write(f"**Interpretation:** The base MRP when fabric amount is zero would be ₹{intercept:.2f}")
            

# Run the app
if __name__ == "__main__":
    main()