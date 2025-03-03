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
            options=["Items", "Stores", "Transaction", "Realtime", "About"],
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

def display_realtime_data_page():
    st.title("Realtime Data Monitor")
    
    # Create tabs for different data types
    tab1, tab2 = st.tabs(["Store Current Amount", "Store Transactions"])
    
    with tab1:
        st.markdown("### Realtime Store Current Amount Data")
        
        # Fetch the latest inventory data from the amount table
        with connect_to_db() as conn:
            import_data = pd.read_sql(
                "SELECT * FROM amount ORDER BY import_time DESC LIMIT 100", 
                conn
            )
        
        if import_data.empty:
            st.info("No Store Current Amount data available.")
        else:
            # Add a refresh button
            if st.button("Refresh Current amount Data"):
                st.rerun()
            
            # Format datetime for better display
            import_data['import_time'] = pd.to_datetime(import_data['import_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Display the data table
            st.dataframe(
                import_data,
                column_config={
                    "id": "ID",
                    "item_id": "Item ID",
                    "amount": st.column_config.NumberColumn("Amount", format="%d"),
                    "import_time": "Import Time",
                    "store_id": "Store ID"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items", f"{import_data['amount'].sum():,}")
            with col2:
                st.metric("Unique Items", f"{import_data['item_id'].nunique():,}")
            with col3:
                st.metric("Stores", f"{import_data['store_id'].nunique():,}")
    
    with tab2:
        st.markdown("### Realtime Transaction Data")
        
        # Fetch the latest transaction data
        with connect_to_db() as conn:
            transaction_data = pd.read_sql(
                "SELECT * FROM transaction ORDER BY buy_time DESC LIMIT 100", 
                conn
            )
        
        if transaction_data.empty:
            st.info("No transaction data available.")
        else:
            # Add a refresh button
            if st.button("Refresh Transaction Data"):
                st.rerun()
            
            # Format datetime for better display
            transaction_data['buy_time'] = pd.to_datetime(transaction_data['buy_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Display the data table
            st.dataframe(
                transaction_data,
                column_config={
                    "id": "ID",
                    "item_id": "Item ID",
                    "amount": st.column_config.NumberColumn("Amount", format="%d"),
                    "buy_time": "Purchase Time",
                    "customer": "Customer ID",
                    "store_id": "Store ID"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Items Sold", f"{transaction_data['amount'].sum():,}")
            with col2:
                st.metric("Unique Customers", f"{transaction_data['customer'].nunique():,}")
            with col3:
                average_order = transaction_data.groupby('id')['amount'].sum().mean()
                st.metric("Avg Order Size", f"{average_order:.1f}")

def display_realtime_visualization_page():
    st.title("Realtime Data Visualization")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Time Series", "Store Comparison", "Prediction"])
    
    # Fetch the data
    with connect_to_db() as conn:
        import_data = pd.read_sql("SELECT * FROM amount", conn)
        transaction_data = pd.read_sql("SELECT * FROM transaction", conn)
    
    if import_data.empty or transaction_data.empty:
        st.info("Not enough data available for visualization.")
        return
        
    # Convert timestamp columns to datetime
    import_data['import_time'] = pd.to_datetime(import_data['import_time'])
    transaction_data['buy_time'] = pd.to_datetime(transaction_data['buy_time'])
    
    with tab1:
        st.markdown("### Time Series Analysis")
        
        # Time aggregation options
        time_grain = st.selectbox(
            "Select Time Granularity",
            options=["Hourly", "Daily", "Weekly", "Monthly"],
            index=1
        )
        
        # Data type selection
        data_type = st.radio(
            "Select Data Type",
            options=["Imports", "Transactions", "Both"],
            horizontal=True
        )
        
        # Prepare time series data based on selection
        if time_grain == "Hourly":
            time_format = "%Y-%m-%d %H:00"
            import_data['time_group'] = import_data['import_time'].dt.strftime(time_format)
            transaction_data['time_group'] = transaction_data['buy_time'].dt.strftime(time_format)
        elif time_grain == "Daily":
            time_format = "%Y-%m-%d"
            import_data['time_group'] = import_data['import_time'].dt.strftime(time_format)
            transaction_data['time_group'] = transaction_data['buy_time'].dt.strftime(time_format)
        elif time_grain == "Weekly":
            import_data['time_group'] = import_data['import_time'].dt.strftime("%Y-%U")
            transaction_data['time_group'] = transaction_data['buy_time'].dt.strftime("%Y-%U")
        else:  # Monthly
            import_data['time_group'] = import_data['import_time'].dt.strftime("%Y-%m")
            transaction_data['time_group'] = transaction_data['buy_time'].dt.strftime("%Y-%m")
        
        # Aggregate by time periods
        import_agg = import_data.groupby('time_group')['amount'].sum().reset_index()
        transaction_agg = transaction_data.groupby('time_group')['amount'].sum().reset_index()
        
        # Create visualization based on selected data type
        if data_type == "Imports":
            fig = px.line(
                import_agg,
                x='time_group',
                y='amount',
                markers=True,
                labels={"time_group": "Time Period", "amount": "Total Amount"},
                title=f"{time_grain} Store Current Amount Amounts"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif data_type == "Transactions":
            fig = px.line(
                transaction_agg,
                x='time_group',
                y='amount',
                markers=True,
                labels={"time_group": "Time Period", "amount": "Total Amount"},
                title=f"{time_grain} Transaction Amounts"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # Both
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=import_agg['time_group'],
                y=import_agg['amount'],
                mode='lines+markers',
                name='Imports'
            ))
            fig.add_trace(go.Scatter(
                x=transaction_agg['time_group'],
                y=transaction_agg['amount'],
                mode='lines+markers',
                name='Transactions'
            ))
            fig.update_layout(
                title=f"{time_grain} Comparison: Imports vs Transactions",
                xaxis_title="Time Period",
                yaxis_title="Total Amount"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Store Comparison")
        
        # Aggregate data by store
        import_by_store = import_data.groupby('store_id')['amount'].sum().reset_index()
        transaction_by_store = transaction_data.groupby('store_id')['amount'].sum().reset_index()
        
        # Create bar chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                import_by_store,
                x='store_id',
                y='amount',
                color='store_id',
                title="Total Imports by Store",
                labels={"store_id": "Store", "amount": "Total Import Amount"}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                transaction_by_store,
                x='store_id',
                y='amount',
                color='store_id',
                title="Total Transactions by Store",
                labels={"store_id": "Store", "amount": "Total Transaction Amount"}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Create a combined view
        combined_data = pd.merge(
            import_by_store, 
            transaction_by_store, 
            on='store_id', 
            suffixes=('_import', '_transaction')
        )
        
        fig3 = px.scatter(
            combined_data,
            x='amount_import',
            y='amount_transaction',
            color='store_id',
            size='amount_import',
            hover_name='store_id',
            title="Imports vs Transactions by Store",
            labels={
                "amount_import": "Total Import Amount", 
                "amount_transaction": "Total Transaction Amount"
            }
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.markdown("### Transaction Amount Prediction")
        
        # Use a simple time series prediction model
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # Need enough data points for prediction
        if len(transaction_data) < 5:
            st.warning("Not enough transaction data for prediction. Need at least 5 data points.")
            return
            
        # Time series prediction
        transaction_daily = transaction_data.groupby(transaction_data['buy_time'].dt.date)['amount'].sum().reset_index()
        transaction_daily = transaction_daily.rename(columns={'buy_time': 'date'})
        
        # Generate numeric features for the dates
        transaction_daily['day_number'] = range(1, len(transaction_daily) + 1)
        
        # Train a polynomial regression model
        X = transaction_daily[['day_number']]
        y = transaction_daily['amount']
        
        # Allow user to select model complexity
        model_complexity = st.slider(
            "Select model complexity (degree of polynomial)",
            min_value=1,
            max_value=5,
            value=2
        )
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=model_complexity)
        X_poly = poly.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Generate prediction for future days
        future_days = st.slider(
            "Number of days to predict",
            min_value=1,
            max_value=30,
            value=7
        )
        
        # Create future date range
        last_date = transaction_daily['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
        
        # Create dataframe for future dates
        future_df = pd.DataFrame({
            'date': future_dates,
            'day_number': range(len(transaction_daily) + 1, len(transaction_daily) + future_days + 1)
        })
        
        # Make predictions
        future_X_poly = poly.transform(future_df[['day_number']])
        future_df['predicted_amount'] = model.predict(future_X_poly)
        
        # Combine actual and predicted data
        full_df = pd.concat([
            transaction_daily[['date', 'amount']],
            future_df[['date', 'predicted_amount']]
        ], axis=0)
        
        # Create chart with actual and predicted values
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=transaction_daily['date'],
            y=transaction_daily['amount'],
            mode='lines+markers',
            name='Actual Transactions'
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=future_df['date'],
            y=future_df['predicted_amount'],
            mode='lines+markers',
            line=dict(dash='dot'),
            marker=dict(symbol='circle-open'),
            name='Predicted Transactions'
        ))
        
        fig.update_layout(
            title="Transaction Amount Prediction",
            xaxis_title="Date",
            yaxis_title="Transaction Amount"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show prediction metrics
        if model_complexity > 1:
            train_predictions = model.predict(X_poly)
            mse = np.mean((train_predictions - y) ** 2)
            rmse = np.sqrt(mse)
            r2 = model.score(X_poly, y)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Squared Error", f"{mse:.2f}")
            col2.metric("Root MSE", f"{rmse:.2f}")
            col3.metric("R² Score", f"{r2:.4f}")
            
            st.markdown("""
            **Note about the model:**
            - This is a polynomial regression model that captures trends in the time series data
            - Higher complexity (polynomial degree) can capture more complex patterns but may overfit
            - The R² score indicates how well the model fits the training data
            - Predictions become less reliable the further into the future they go
            """)

def display_realtime_statistics_page():
    st.title("Realtime Statistics")
    
    # Fetch the data
    with connect_to_db() as conn:
        import_data = pd.read_sql("SELECT * FROM amount", conn)
        transaction_data = pd.read_sql("SELECT * FROM transaction", conn)
    
    if import_data.empty or transaction_data.empty:
        st.info("Not enough data available for statistics.")
        return
    
    # Create tabs for different statistics
    tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Time Analysis", "Correlation Analysis"])
    
    with tab1:
        st.markdown("### Summary Statistics")
        
        # Import data statistics
        st.subheader("Store Current Amount Statistics")
        import_stats = import_data['amount'].describe()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Imports", f"{import_data['amount'].sum():,}")
        col2.metric("Average Import Amount", f"{import_data['amount'].mean():.2f}")
        col3.metric("Min Import Amount", f"{import_data['amount'].min():,}")
        col4.metric("Max Import Amount", f"{import_data['amount'].max():,}")
        
        # Transaction data statistics
        st.subheader("Transaction Statistics")
        transaction_stats = transaction_data['amount'].describe()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sales", f"{transaction_data['amount'].sum():,}")
        col2.metric("Average Transaction Amount", f"{transaction_data['amount'].mean():.2f}")
        col3.metric("Min Transaction Amount", f"{transaction_data['amount'].min():,}")
        col4.metric("Max Transaction Amount", f"{transaction_data['amount'].max():,}")
        
        # Store statistics
        st.subheader("Store Statistics")
        
        # Calculate store efficiency (transaction amount / import amount)
        store_stats = pd.merge(
            import_data.groupby('store_id')['amount'].sum().reset_index().rename(columns={'amount': 'import_amount'}),
            transaction_data.groupby('store_id')['amount'].sum().reset_index().rename(columns={'amount': 'transaction_amount'})
        )
        
        store_stats['efficiency'] = (store_stats['transaction_amount'] / store_stats['import_amount'] * 100).round(2)
        
        st.dataframe(
            store_stats,
            column_config={
                "store_id": "Store ID",
                "import_amount": st.column_config.NumberColumn("Total Imports", format="%d"),
                "transaction_amount": st.column_config.NumberColumn("Total Sales", format="%d"),
                "efficiency": st.column_config.ProgressColumn("Efficiency (%)", format="%.2f%%", min_value=0, max_value=100)
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Best performing store
        best_store = store_stats.loc[store_stats['efficiency'].idxmax()]
        
        st.info(f"**Best performing store:** {best_store['store_id']} with {best_store['efficiency']}% efficiency")
    
    with tab2:
        st.markdown("### Time-based Analysis")
        
        # Convert timestamps to datetime
        import_data['import_time'] = pd.to_datetime(import_data['import_time'])
        transaction_data['buy_time'] = pd.to_datetime(transaction_data['buy_time'])
        
        # Add time components
        import_data['hour'] = import_data['import_time'].dt.hour
        import_data['day'] = import_data['import_time'].dt.day_name()
        import_data['month'] = import_data['import_time'].dt.month_name()
        
        transaction_data['hour'] = transaction_data['buy_time'].dt.hour
        transaction_data['day'] = transaction_data['buy_time'].dt.day_name()
        transaction_data['month'] = transaction_data['buy_time'].dt.month_name()
        
        # Time period selection
        time_period = st.selectbox(
            "Select Time Period for Analysis",
            options=["Hour of Day", "Day of Week", "Month"]
        )
        
        if time_period == "Hour of Day":
            period_col = 'hour'
            sort_order = True  # numeric sort
        elif time_period == "Day of Week":
            period_col = 'day'
            # Custom sort order for days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sort_order = False  # custom sort
        else:  # Month
            period_col = 'month'
            # Custom sort order for months
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            sort_order = False  # custom sort
        
        # Aggregate data by selected period
        import_by_period = import_data.groupby(period_col)['amount'].sum().reset_index()
        transaction_by_period = transaction_data.groupby(period_col)['amount'].sum().reset_index()
        
        # Sort based on period type
        if sort_order:
            import_by_period = import_by_period.sort_values(period_col)
            transaction_by_period = transaction_by_period.sort_values(period_col)
        else:
            if period_col == 'day':
                import_by_period['sort_order'] = import_by_period[period_col].map({day: i for i, day in enumerate(day_order)})
                transaction_by_period['sort_order'] = transaction_by_period[period_col].map({day: i for i, day in enumerate(day_order)})
            else:  # month
                import_by_period['sort_order'] = import_by_period[period_col].map({month: i for i, month in enumerate(month_order)})
                transaction_by_period['sort_order'] = transaction_by_period[period_col].map({month: i for i, month in enumerate(month_order)})
            
            import_by_period = import_by_period.sort_values('sort_order')
            transaction_by_period = transaction_by_period.sort_values('sort_order')
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                import_by_period,
                x=period_col,
                y='amount',
                title=f"Imports by {time_period}",
                labels={period_col: time_period, "amount": "Total Import Amount"}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                transaction_by_period,
                x=period_col,
                y='amount',
                title=f"Transactions by {time_period}",
                labels={period_col: time_period, "amount": "Total Transaction Amount"}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Peak time analysis
        peak_import_time = import_by_period.loc[import_by_period['amount'].idxmax()][period_col]
        peak_transaction_time = transaction_by_period.loc[transaction_by_period['amount'].idxmax()][period_col]
        
        st.markdown(f"**Peak import time:** {peak_import_time}")
        st.markdown(f"**Peak transaction time:** {peak_transaction_time}")
        
        if peak_import_time == peak_transaction_time:
            st.success(f"Both imports and transactions peak during the same {time_period.lower()} ({peak_import_time}).")
        else:
            st.warning(f"Import and transaction peaks occur at different times. Consider adjusting inventory timing.")
    
    with tab3:
        st.markdown("### Correlation Analysis")
        
        # Join import and transaction data by item_id and store_id
        merged_data = pd.merge(
            import_data.groupby(['item_id', 'store_id'])['amount'].sum().reset_index().rename(columns={'amount': 'import_amount'}),
            transaction_data.groupby(['item_id', 'store_id'])['amount'].sum().reset_index().rename(columns={'amount': 'transaction_amount'}),
            on=['item_id', 'store_id'],
            how='outer'
        ).fillna(0)
        
        # Create scatter plot of import vs transaction amounts
        fig = px.scatter(
            merged_data,
            x='import_amount',
            y='transaction_amount',
            color='store_id',
            hover_data=['item_id'],
            trendline='ols',
            title="Correlation: Import Amount vs Transaction Amount",
            labels={
                'import_amount': 'Import Amount',
                'transaction_amount': 'Transaction Amount'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation coefficient
        corr = merged_data['import_amount'].corr(merged_data['transaction_amount'])
        
        # Show correlation value
        st.metric("Correlation Coefficient", f"{corr:.4f}")
        
        if corr > 0.7:
            st.success("Strong positive correlation: Import amounts strongly predict transaction amounts.")
        elif corr > 0.3:
            st.info("Moderate positive correlation: Import amounts moderately predict transaction amounts.")
        else:
            st.warning("Weak correlation: Import amounts don't strongly predict transaction amounts.")
        
        # Item-level analysis
        st.subheader("Item Efficiency Analysis")
        
        # Calculate efficiency ratio
        merged_data['efficiency'] = np.where(
            merged_data['import_amount'] > 0,
            (merged_data['transaction_amount'] / merged_data['import_amount'] * 100).round(2),
            0
        )
        
        # Sort by efficiency
        top_items = merged_data.sort_values('efficiency', ascending=False).head(10)
        
        st.markdown("**Top 10 Most Efficient Items**")
        st.dataframe(
            top_items,
            column_config={
                "item_id": "Item ID",
                "store_id": "Store ID",
                "import_amount": st.column_config.NumberColumn("Import Amount", format="%d"),
                "transaction_amount": st.column_config.NumberColumn("Transaction Amount", format="%d"),
                "efficiency": st.column_config.ProgressColumn("Efficiency (%)", format="%.2f%%", min_value=0, max_value=100)
            },
            hide_index=True,
            use_container_width=True
        )

def display_transaction_data_page():
    st.title("Transaction Data")
    
    # Initialize session state variables if they don't exist
    if 'transaction_page_num' not in st.session_state:
        st.session_state.transaction_page_num = 0
    if 'transaction_items_per_page' not in st.session_state:
        st.session_state.transaction_items_per_page = 20
    if 'transaction_filters' not in st.session_state:
        st.session_state.transaction_filters = {}
    
    # Create layout with columns
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Filters")
        st.markdown("---")
        
        # Connect to database to get filter options
        with connect_to_db() as conn:
            # Get distinct store IDs
            stores = pd.read_sql("SELECT DISTINCT store_id FROM item_sales ORDER BY store_id", conn)['store_id'].tolist()
            
            # Get distinct item IDs
            items = pd.read_sql("SELECT DISTINCT item_id FROM item_sales ORDER BY item_id", conn)['item_id'].tolist()
            
            # Get min/max values for numeric columns
            ranges = pd.read_sql("""
                SELECT 
                    MIN(item_visibility) as min_visibility, 
                    MAX(item_visibility) as max_visibility,
                    MIN(item_store_sales) as min_sales,
                    MAX(item_store_sales) as max_sales
                FROM item_sales
            """, conn).iloc[0].to_dict()
        
        # Filtering widgets
        selected_stores = st.multiselect(
            "Store ID", 
            options=stores,
            default=st.session_state.transaction_filters.get('stores', [])
        )
        
        selected_items = st.multiselect(
            "Item ID", 
            options=items,
            default=st.session_state.transaction_filters.get('items', [])
        )
        
        visibility_range = st.slider(
            "Item Visibility Range",
            min_value=float(ranges['min_visibility']),
            max_value=float(ranges['max_visibility']),
            value=(
                st.session_state.transaction_filters.get('min_visibility', float(ranges['min_visibility'])),
                st.session_state.transaction_filters.get('max_visibility', float(ranges['max_visibility']))
            ),
            step=0.001
        )
        
        sales_range = st.slider(
            "Sales Range",
            min_value=float(ranges['min_sales']),
            max_value=float(ranges['max_sales']),
            value=(
                st.session_state.transaction_filters.get('min_sales', float(ranges['min_sales'])),
                st.session_state.transaction_filters.get('max_sales', float(ranges['max_sales']))
            ),
            step=1000.0
        )
        
        # Update filters in session state
        filters = {}
        if selected_stores:
            filters['stores'] = selected_stores
        if selected_items:
            filters['items'] = selected_items
        
        filters['min_visibility'] = visibility_range[0]
        filters['max_visibility'] = visibility_range[1]
        filters['min_sales'] = sales_range[0]
        filters['max_sales'] = sales_range[1]
        
        # Apply filters button
        if st.button("Apply Filters"):
            st.session_state.transaction_filters = filters
            st.session_state.transaction_page_num = 0  # Reset to first page
            st.rerun()
        
        # Reset filters button
        if st.button("Reset Filters"):
            st.session_state.transaction_filters = {}
            st.session_state.transaction_page_num = 0  # Reset to first page
            st.rerun()
    
    with col1:
        # Fetch transaction data with pagination and filtering
        transactions_df, total_transactions = fetch_transactions(
            page=st.session_state.transaction_page_num,
            items_per_page=st.session_state.transaction_items_per_page,
            filters=st.session_state.transaction_filters
        )
        
        # Display total transaction count and current range
        total_pages = (total_transactions - 1) // st.session_state.transaction_items_per_page + 1
        start_item = st.session_state.transaction_page_num * st.session_state.transaction_items_per_page + 1
        end_item = min(start_item + st.session_state.transaction_items_per_page - 1, total_transactions)
        
        st.markdown(f"Showing {start_item} to {end_item} of {total_transactions} transactions")
        
        # Display the transactions table
        if not transactions_df.empty:
            # Create an interactive table with Streamlit
            st.dataframe(
                transactions_df,
                column_config={
                    "item_id": "Item ID",
                    "store_id": "Store ID",
                    "item_visibility": st.column_config.NumberColumn("Item Visibility", format="%.6f"),
                    "item_store_sales": st.column_config.NumberColumn("Sales", format="₹%d"),
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Add pagination controls
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.session_state.transaction_page_num > 0:
                    if st.button("← Previous"):
                        st.session_state.transaction_page_num -= 1
                        st.rerun()
            with col2:
                st.markdown(f"<div class='center-align'>Page {st.session_state.transaction_page_num + 1} of {total_pages}</div>", unsafe_allow_html=True)
            with col3:
                if st.session_state.transaction_page_num < total_pages - 1:
                    if st.button("Next →"):
                        st.session_state.transaction_page_num += 1
                        st.rerun()
        else:
            st.info("No transactions found matching your criteria.")

# Function to fetch transactions with filters for the data page
def fetch_transactions(page=0, items_per_page=20, filters=None):
    """Fetch transactions from database with pagination and filtering."""
    offset = page * items_per_page
    
    # Base query
    query = """
        SELECT 
            item_id, item_visibility, store_id, item_store_sales
        FROM item_sales
    """
    
    # Add filters if provided
    where_clauses = []
    params = []
    
    if filters:
        if filters.get('stores'):
            placeholders = ', '.join(['%s'] * len(filters['stores']))
            where_clauses.append(f"store_id IN ({placeholders})")
            params.extend(filters['stores'])
            
        if filters.get('items'):
            placeholders = ', '.join(['%s'] * len(filters['items']))
            where_clauses.append(f"item_id IN ({placeholders})")
            params.extend(filters['items'])
            
        if filters.get('min_visibility') is not None:
            where_clauses.append("item_visibility >= %s")
            params.append(filters['min_visibility'])
            
        if filters.get('max_visibility') is not None:
            where_clauses.append("item_visibility <= %s")
            params.append(filters['max_visibility'])
            
        if filters.get('min_sales') is not None:
            where_clauses.append("item_store_sales >= %s")
            params.append(filters['min_sales'])
            
        if filters.get('max_sales') is not None:
            where_clauses.append("item_store_sales <= %s")
            params.append(filters['max_sales'])
    
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    
    # Add sorting
    query += " ORDER BY store_id, item_id"
    
    # Add pagination
    query += f" LIMIT {items_per_page} OFFSET {offset}"
    
    with connect_to_db() as conn:
        # Get transactions
        df = pd.read_sql(query, conn, params=params)
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM item_sales"
        if where_clauses:
            count_query += " WHERE " + " AND ".join(where_clauses)
        total_items = pd.read_sql(count_query, conn, params=params).iloc[0, 0]
        
    return df, total_items

def display_transaction_visualization_page():
    st.title("Transaction Visualization")
    
    # Fetch all transactions for visualization
    with connect_to_db() as conn:
        transactions_df = pd.read_sql("SELECT * FROM item_sales", conn)
    
    if transactions_df.empty:
        st.info("No transaction data found in the database.")
        return
    
    # Create tabs for different visualization perspectives
    tab1, tab2, tab3 = st.tabs(["Sales Analysis", "Visibility Analysis", "Store Performance"])
    
    with tab1:
        st.markdown("### Sales Analysis")
        
        # Sales visualization options
        viz_type = st.selectbox(
            "Select Visualization Type",
            options=["Top Items by Sales", "Sales Distribution by Store", "Sales Trend Analysis"],
            key="sales_viz_type"
        )
        
        if viz_type == "Top Items by Sales":
            # Aggregate sales by item
            top_n = st.slider("Number of top items to show", 5, 20, 10, key="top_items_slider")
            
            item_sales = transactions_df.groupby('item_id')['item_store_sales'].sum().reset_index()
            item_sales = item_sales.sort_values('item_store_sales', ascending=False).head(top_n)
            
            fig = px.bar(
                item_sales,
                x='item_id',
                y='item_store_sales',
                title=f"Top {top_n} Items by Total Sales",
                labels={
                    'item_id': 'Item ID',
                    'item_store_sales': 'Total Sales (₹)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.markdown("#### Key Insights")
            st.write(f"- The top selling item generated ₹{item_sales['item_store_sales'].max():,.2f} in sales")
            st.write(f"- The top {top_n} items account for {item_sales['item_store_sales'].sum() / transactions_df['item_store_sales'].sum() * 100:.2f}% of total sales")
        
        elif viz_type == "Sales Distribution by Store":
            # Aggregate sales by store
            store_sales = transactions_df.groupby('store_id')['item_store_sales'].sum().reset_index()
            store_sales = store_sales.sort_values('item_store_sales', ascending=False)
            
            # Add visualization options
            chart_type = st.radio("Chart Type", ["Bar Chart", "Pie Chart"], horizontal=True)
            
            if chart_type == "Bar Chart":
                fig = px.bar(
                    store_sales,
                    x='store_id',
                    y='item_store_sales',
                    title="Sales Distribution by Store",
                    labels={
                        'store_id': 'Store ID',
                        'item_store_sales': 'Total Sales (₹)'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.pie(
                    store_sales,
                    values='item_store_sales',
                    names='store_id',
                    title="Sales Distribution by Store"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.markdown("#### Store Performance Insights")
            st.write(f"- Highest performing store: {store_sales.iloc[0]['store_id']} with ₹{store_sales.iloc[0]['item_store_sales']:,.2f} in sales")
            st.write(f"- Lowest performing store: {store_sales.iloc[-1]['store_id']} with ₹{store_sales.iloc[-1]['item_store_sales']:,.2f} in sales")
            st.write(f"- The difference between highest and lowest performing stores is ₹{store_sales.iloc[0]['item_store_sales'] - store_sales.iloc[-1]['item_store_sales']:,.2f}")
        
        elif viz_type == "Sales Trend Analysis":
            st.info("This visualization would typically show sales trends over time. The current dataset doesn't appear to include time data, but in a real system, you would add date/time filtering options here.")
            
            # Create a simulated trend for demonstration
            st.markdown("#### Simulated Sales Trend (For Demonstration)")
            
            # Create a sample of unique items and stores for simulation
            items = transactions_df['item_id'].unique()[:5]
            
            # Create simulated time series data
            np.random.seed(42)  # For reproducibility
            dates = pd.date_range(start='2025-01-01', periods=12, freq='M')
            simulated_data = []
            
            for item in items:
                base_sales = np.mean(transactions_df[transactions_df['item_id'] == item]['item_store_sales'])
                for date in dates:
                    # Add some randomness and a slight upward trend
                    trend_factor = 1 + (date.month / 12) * 0.2
                    randomness = np.random.normal(1, 0.1)
                    simulated_data.append({
                        'date': date,
                        'item_id': item,
                        'sales': base_sales * trend_factor * randomness
                    })
            
            trend_df = pd.DataFrame(simulated_data)
            
            # Visualization of simulated data
            fig = px.line(
                trend_df,
                x='date',
                y='sales',
                color='item_id',
                title="Simulated Monthly Sales Trend by Item (2025)",
                labels={
                    'date': 'Month',
                    'sales': 'Sales (₹)',
                    'item_id': 'Item ID'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Note: This is simulated data for visualization purposes only.")
    
    with tab2:
        st.markdown("### Visibility Analysis")
        
        # Group data by item and calculate average visibility and total sales
        item_visibility_sales = transactions_df.groupby('item_id').agg({
            'item_visibility': 'mean',
            'item_store_sales': 'sum'
        }).reset_index()
        
        # Create scatter plot of visibility vs sales
        fig = px.scatter(
            item_visibility_sales,
            x='item_visibility',
            y='item_store_sales',
            hover_name='item_id',
            title="Relationship Between Item Visibility and Sales",
            labels={
                'item_visibility': 'Average Item Visibility',
                'item_store_sales': 'Total Sales (₹)'
            },
            trendline='ols'  # Add trend line
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        correlation = item_visibility_sales['item_visibility'].corr(item_visibility_sales['item_store_sales'])
        
        # Display correlation and insights
        st.markdown("#### Visibility-Sales Correlation")
        st.metric("Correlation Coefficient", f"{correlation:.4f}")
        
        if correlation > 0.7:
            st.success("Strong positive correlation: Higher visibility is strongly associated with higher sales")
        elif correlation > 0.3:
            st.info("Moderate positive correlation: Higher visibility tends to be associated with higher sales")
        elif correlation > -0.3:
            st.warning("Weak or no correlation: Visibility and sales don't show a clear relationship")
        else:
            st.error("Negative correlation: Items with higher visibility tend to have lower sales (unusual)")
        
        # Visibility distribution
        st.markdown("#### Visibility Distribution")
        fig = px.histogram(
            transactions_df,
            x='item_visibility',
            nbins=20,
            title="Distribution of Item Visibility",
            labels={
                'item_visibility': 'Item Visibility',
                'count': 'Number of Transactions'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.markdown("#### Key Insights")
        st.write(f"- Average visibility across all transactions: {transactions_df['item_visibility'].mean():.6f}")
        st.write(f"- Items with highest visibility have {transactions_df.nlargest(5, 'item_visibility')['item_store_sales'].mean() / transactions_df.nsmallest(5, 'item_visibility')['item_store_sales'].mean():.2f}x the sales of items with lowest visibility")
    
    with tab3:
        st.markdown("### Store Performance")
        
        # Metrics by store
        store_metrics = transactions_df.groupby('store_id').agg({
            'item_store_sales': ['sum', 'mean', 'count'],
            'item_visibility': 'mean'
        })
        store_metrics.columns = ['total_sales', 'avg_sales_per_item', 'item_count', 'avg_visibility']
        store_metrics = store_metrics.reset_index()
        
        # Sort options
        sort_by = st.selectbox(
            "Sort Stores By",
            options=["Total Sales", "Average Sales Per Item", "Number of Items", "Average Visibility"],
            index=0
        )
        
        # Map selection to column name
        sort_map = {
            "Total Sales": "total_sales",
            "Average Sales Per Item": "avg_sales_per_item",
            "Number of Items": "item_count",
            "Average Visibility": "avg_visibility"
        }
        
        # Sort data
        sorted_metrics = store_metrics.sort_values(sort_map[sort_by], ascending=False)
        
        # Display top/bottom stores
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### Top 5 Stores by {sort_by}")
            st.dataframe(
                sorted_metrics.head(5),
                column_config={
                    "store_id": "Store ID",
                    "total_sales": st.column_config.NumberColumn("Total Sales", format="₹%d"),
                    "avg_sales_per_item": st.column_config.NumberColumn("Avg Sales/Item", format="₹%d"),
                    "item_count": st.column_config.NumberColumn("Item Count", format="%d"),
                    "avg_visibility": st.column_config.NumberColumn("Avg Visibility", format="%.6f"),
                },
                hide_index=True,
                use_container_width=True
            )
        
        with col2:
            st.markdown(f"#### Bottom 5 Stores by {sort_by}")
            st.dataframe(
                sorted_metrics.tail(5),
                column_config={
                    "store_id": "Store ID",
                    "total_sales": st.column_config.NumberColumn("Total Sales", format="₹%d"),
                    "avg_sales_per_item": st.column_config.NumberColumn("Avg Sales/Item", format="₹%d"),
                    "item_count": st.column_config.NumberColumn("Item Count", format="%d"),
                    "avg_visibility": st.column_config.NumberColumn("Avg Visibility", format="%.6f"),
                },
                hide_index=True,
                use_container_width=True
            )
        
        # Visualization of all stores
        st.markdown("#### All Stores Performance")
        
        # Create bar chart of stores by selected metric
        fig = px.bar(
            sorted_metrics,
            x='store_id',
            y=sort_map[sort_by],
            title=f"Stores Ranked by {sort_by}",
            labels={
                'store_id': 'Store ID',
                sort_map[sort_by]: sort_by
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional store analysis - Bubble chart
        st.markdown("#### Store Multi-dimensional Analysis")
        st.write("This bubble chart shows the relationship between total sales, average visibility, and number of items for each store:")
        
        fig = px.scatter(
            store_metrics,
            x='avg_visibility',
            y='total_sales',
            size='item_count',
            hover_name='store_id',
            title="Store Performance: Sales vs. Visibility vs. Item Count",
            labels={
                'avg_visibility': 'Average Item Visibility',
                'total_sales': 'Total Sales (₹)',
                'item_count': 'Number of Items'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

def display_transaction_statistics_page():
    st.title("Transaction Statistics")
    
    # Fetch all transactions for statistics
    with connect_to_db() as conn:
        transactions_df = pd.read_sql("SELECT * FROM item_sales", conn)
    
    if transactions_df.empty:
        st.info("No transaction data found in the database.")
        return
    
    # Create tabs for different statistics views
    tab1, tab2, tab3 = st.tabs(["Overall Statistics", "Item Analysis", "Store Analysis"])
    
    with tab1:
        st.markdown("### Overall Transaction Statistics")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = transactions_df['item_store_sales'].sum()
            st.metric("Total Sales", f"₹{total_sales:,.2f}")
        
        with col2:
            avg_transaction = transactions_df['item_store_sales'].mean()
            st.metric("Average Transaction", f"₹{avg_transaction:,.2f}")
        
        with col3:
            total_transactions = len(transactions_df)
            st.metric("Total Transactions", f"{total_transactions:,}")
        
        with col4:
            avg_visibility = transactions_df['item_visibility'].mean()
            st.metric("Average Visibility", f"{avg_visibility:.6f}")
        
        # Summary statistics for numerical columns
        st.markdown("#### Numerical Data Summary")
        
        numeric_cols = ['item_visibility', 'item_store_sales']
        numeric_stats = transactions_df[numeric_cols].describe().T
        
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
        
        # Distribution of sales
        st.markdown("#### Sales Distribution")
        
        fig = px.histogram(
            transactions_df,
            x='item_store_sales',
            nbins=20,
            title="Distribution of Sales",
            labels={'item_store_sales': 'Sales (₹)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution of visibility
        st.markdown("#### Visibility Distribution")
        
        fig = px.histogram(
            transactions_df,
            x='item_visibility',
            nbins=20,
            title="Distribution of Item Visibility",
            labels={'item_visibility': 'Visibility'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Item-Based Analysis")
        
        # Aggregate statistics by item
        item_stats = transactions_df.groupby('item_id').agg({
            'item_store_sales': ['sum', 'mean', 'count', 'std'],
            'item_visibility': ['mean', 'min', 'max', 'std']
        })
        
        # Flatten the multi-level columns
        item_stats.columns = ['_'.join(col).strip() for col in item_stats.columns.values]
        item_stats = item_stats.reset_index()
        
        # Show detailed item statistics
        st.markdown("#### Item Performance Statistics")
        
        st.dataframe(
            item_stats,
            column_config={
                "item_id": "Item ID",
                "item_store_sales_sum": st.column_config.NumberColumn("Total Sales", format="₹%d"),
                "item_store_sales_mean": st.column_config.NumberColumn("Avg Sales/Store", format="₹%d"),
                "item_store_sales_count": st.column_config.NumberColumn("Store Count", format="%d"),
                "item_store_sales_std": st.column_config.NumberColumn("Sales Std Dev", format="₹%d"),
                "item_visibility_mean": st.column_config.NumberColumn("Avg Visibility", format="%.6f"),
                "item_visibility_min": st.column_config.NumberColumn("Min Visibility", format="%.6f"),
                "item_visibility_max": st.column_config.NumberColumn("Max Visibility", format="%.6f"),
                "item_visibility_std": st.column_config.NumberColumn("Visibility Std Dev", format="%.6f"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Item statistics summary
        st.markdown("#### Top Performing Items")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Top 5 Items by Total Sales")
            top_items_sales = item_stats.nlargest(5, 'item_store_sales_sum')[['item_id', 'item_store_sales_sum']]
            
            fig = px.bar(
                top_items_sales,
                x='item_id',
                y='item_store_sales_sum',
                title="Top 5 Items by Total Sales",
                labels={
                    'item_id': 'Item ID',
                    'item_store_sales_sum': 'Total Sales (₹)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Top 5 Items by Average Sales per Store")
            top_items_avg = item_stats.nlargest(5, 'item_store_sales_mean')[['item_id', 'item_store_sales_mean']]
            
            fig = px.bar(
                top_items_avg,
                x='item_id',
                y='item_store_sales_mean',
                title="Top 5 Items by Avg Sales/Store",
                labels={
                    'item_id': 'Item ID',
                    'item_store_sales_mean': 'Avg Sales/Store (₹)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sales and visibility relationship by item
        st.markdown("#### Sales and Visibility Relationship by Item")
        
        fig = px.scatter(
            item_stats,
            x='item_visibility_mean',
            y='item_store_sales_mean',
            hover_name='item_id',
            size='item_store_sales_sum',
            title="Item Performance: Average Sales vs. Average Visibility",
            labels={
                'item_visibility_mean': 'Average Visibility',
                'item_store_sales_mean': 'Average Sales/Store (₹)',
                'item_store_sales_sum': 'Total Sales'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.markdown("#### Item Stats Correlation Analysis")
        
        # Calculate correlation matrix
        corr_cols = ['item_store_sales_sum', 'item_store_sales_mean', 'item_store_sales_std', 
                     'item_visibility_mean', 'item_visibility_std']
        corr_matrix = item_stats[corr_cols].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix of Item Metrics",
            labels=dict(color="Correlation"),
            x=[col.replace('item_', '').replace('_', ' ').title() for col in corr_cols],
            y=[col.replace('item_', '').replace('_', ' ').title() for col in corr_cols]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Store-Based Analysis")
        
        # Aggregate statistics by store
        store_stats = transactions_df.groupby('store_id').agg({
            'item_store_sales': ['sum', 'mean', 'count', 'std'],
            'item_visibility': ['mean', 'min', 'max', 'std']
        })
        
        # Flatten the multi-level columns
        store_stats.columns = ['_'.join(col).strip() for col in store_stats.columns.values]
        store_stats = store_stats.reset_index()
        
        # Show detailed store statistics
        st.markdown("#### Store Performance Statistics")
        
        st.dataframe(
            store_stats,
            column_config={
                "store_id": "Store ID",
                "item_store_sales_sum": st.column_config.NumberColumn("Total Sales", format="₹%d"),
                "item_store_sales_mean": st.column_config.NumberColumn("Avg Sales/Item", format="₹%d"),
                "item_store_sales_count": st.column_config.NumberColumn("Item Count", format="%d"),
                "item_store_sales_std": st.column_config.NumberColumn("Sales Std Dev", format="₹%d"),
                "item_visibility_mean": st.column_config.NumberColumn("Avg Visibility", format="%.6f"),
                "item_visibility_min": st.column_config.NumberColumn("Min Visibility", format="%.6f"),
                "item_visibility_max": st.column_config.NumberColumn("Max Visibility", format="%.6f"),
                "item_visibility_std": st.column_config.NumberColumn("Visibility Std Dev", format="%.6f"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Store statistics summary
        st.markdown("#### Store Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Top 5 Stores by Total Sales")
            top_stores_sales = store_stats.nlargest(5, 'item_store_sales_sum')[['store_id', 'item_store_sales_sum']]
            
            fig = px.bar(
                top_stores_sales,
                x='store_id',
                y='item_store_sales_sum',
                title="Top 5 Stores by Total Sales",
                labels={
                    'store_id': 'Store ID',
                    'item_store_sales_sum': 'Total Sales (₹)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Top 5 Stores by Average Sales per Item")
            top_stores_avg = store_stats.nlargest(5, 'item_store_sales_mean')[['store_id', 'item_store_sales_mean']]
            
            fig = px.bar(
                top_stores_avg,
                x='store_id',
                y='item_store_sales_mean',
                title="Top 5 Stores by Avg Sales/Item",
                labels={
                    'store_id': 'Store ID',
                    'item_store_sales_mean': 'Avg Sales/Item (₹)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Store performance distribution
        st.markdown("#### Store Performance Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                store_stats,
                x='item_store_sales_sum',
                nbins=15,
                title="Distribution of Total Sales Across Stores",
                labels={'item_store_sales_sum': 'Total Sales (₹)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                store_stats,
                x='item_store_sales_mean',
                nbins=15,
                title="Distribution of Average Sales per Item Across Stores",
                labels={'item_store_sales_mean': 'Average Sales per Item (₹)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Store efficiency analysis
        st.markdown("#### Store Efficiency Analysis")
        
        # Calculate efficiency metrics
        store_stats['sales_per_visibility'] = store_stats['item_store_sales_mean'] / store_stats['item_visibility_mean']
        
        # Sort by efficiency
        efficient_stores = store_stats.sort_values('sales_per_visibility', ascending=False)
        
        st.write("This chart shows how effectively stores are converting visibility into sales:")
        
        fig = px.bar(
            efficient_stores.head(10),
            x='store_id',
            y='sales_per_visibility',
            title="Top 10 Stores by Sales/Visibility Ratio",
            labels={
                'store_id': 'Store ID',
                'sales_per_visibility': 'Sales per Visibility Unit (₹)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Stores with higher sales per visibility unit are more efficient at converting item visibility into actual sales.")


# ---- STORES DATA PAGE ----
def display_stores_data_page():
    st.title("Stores Management")
    
    # Initialize session state variables if they don't exist
    if 'stores_page_num' not in st.session_state:
        st.session_state.stores_page_num = 0
    if 'stores_per_page' not in st.session_state:
        st.session_state.stores_per_page = 20
    if 'stores_editing' not in st.session_state:
        st.session_state.stores_editing = False
    if 'edit_store' not in st.session_state:
        st.session_state.edit_store = {}
    if 'confirming_store_delete' not in st.session_state:
        st.session_state.confirming_store_delete = None
    if 'adding_new_store' not in st.session_state:
        st.session_state.adding_new_store = False
    if 'stores_filters' not in st.session_state:
        st.session_state.stores_filters = {}
    
    # Create layout with columns
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Filters")
        st.markdown("---")
        
        # Get distinct values for filtering
        store_sizes = ["Small", "Medium", "Large"]  # Assuming these are the standard sizes
        location_types = ["Region 1", "Region 2", "Region 3"]  # Example regions
        store_types = ["Mall Type1", "Mall Type2", "Standalone"]  # Example store types
        
        # Filtering widgets
        selected_sizes = st.multiselect(
            "Store Size", 
            options=store_sizes,
            default=st.session_state.stores_filters.get('store_sizes', [])
        )
        
        selected_locations = st.multiselect(
            "Location Type", 
            options=location_types,
            default=st.session_state.stores_filters.get('location_types', [])
        )
        
        selected_store_types = st.multiselect(
            "Store Type", 
            options=store_types,
            default=st.session_state.stores_filters.get('store_types', [])
        )
        
        establishment_range = st.slider(
            "Establishment Year Range",
            min_value=1980,
            max_value=2025,
            value=(
                st.session_state.stores_filters.get('min_year', 1980),
                st.session_state.stores_filters.get('max_year', 2025)
            ),
            step=1
        )
        
        # Update filters in session state
        filters = {}
        if selected_sizes:
            filters['store_sizes'] = selected_sizes
        if selected_locations:
            filters['location_types'] = selected_locations
        if selected_store_types:
            filters['store_types'] = selected_store_types
        
        filters['min_year'] = establishment_range[0]
        filters['max_year'] = establishment_range[1]
        
        # Apply filters button
        if st.button("Apply Filters"):
            st.session_state.stores_filters = filters
            st.session_state.stores_page_num = 0  # Reset to first page
            st.rerun()
        
        # Reset filters button
        if st.button("Reset Filters"):
            st.session_state.stores_filters = {}
            st.session_state.stores_page_num = 0  # Reset to first page
            st.rerun()
    
    with col1:
        # Add new store button
        if not st.session_state.adding_new_store and not st.session_state.stores_editing:
            if st.button("Add New Store"):
                st.session_state.adding_new_store = True
                st.rerun()
        
        # Form for adding new store
        if st.session_state.adding_new_store:
            st.markdown("### Add New Store")
            st.markdown("---")
            
            with st.form(key="add_store_form"):
                new_store_id = st.text_input("Store ID", key="new_store_id")
                new_establishment_year = st.number_input("Establishment Year", min_value=1900, max_value=2025, value=2022, key="new_est_year")
                new_store_size = st.selectbox("Store Size", options=store_sizes, key="new_size")
                new_location_type = st.selectbox("Location Type", options=location_types, key="new_location")
                new_store_type = st.selectbox("Store Type", options=store_types, key="new_store_type")
                
                col1, col2 = st.columns(2)
                submitted = col1.form_submit_button("Save")
                if col2.form_submit_button("Cancel"):
                    st.session_state.adding_new_store = False
                    st.rerun()
                
                if submitted:
                    if not new_store_id:
                        st.error("Store ID is required")
                    else:
                        # Add code to insert store into database
                        st.session_state.adding_new_store = False
                        st.success(f"Store {new_store_id} added successfully!")
                        time.sleep(1)
                        st.rerun()
        
        # Form for editing store
        if st.session_state.stores_editing:
            st.markdown(f"### Edit Store: {st.session_state.edit_store['store_id']}")
            st.markdown("---")
            
            with st.form(key="edit_store_form"):
                edit_store = st.session_state.edit_store
                edit_establishment_year = st.number_input("Establishment Year", value=int(edit_store['store_establishment_year']), min_value=1900, max_value=2025)
                edit_store_size = st.selectbox("Store Size", options=store_sizes, index=store_sizes.index(edit_store['store_size']) if edit_store['store_size'] in store_sizes else 0)
                edit_location_type = st.selectbox("Location Type", options=location_types, index=location_types.index(edit_store['store_location_type']) if edit_store['store_location_type'] in location_types else 0)
                edit_store_type = st.selectbox("Store Type", options=store_types, index=store_types.index(edit_store['store_type']) if edit_store['store_type'] in store_types else 0)
                
                col1, col2 = st.columns(2)
                submitted = col1.form_submit_button("Update")
                if col2.form_submit_button("Cancel"):
                    st.session_state.stores_editing = False
                    st.rerun()
                
                if submitted:
                    # Add code to update store in database
                    st.session_state.stores_editing = False
                    st.success(f"Store {edit_store['store_id']} updated successfully!")
                    time.sleep(1)
                    st.rerun()
        
        # Delete confirmation
        if st.session_state.confirming_store_delete:
            st.markdown(f"### Confirm Delete: {st.session_state.confirming_store_delete}")
            st.markdown("---")
            st.warning("Are you sure you want to delete this store? This action cannot be undone.")
            
            col1, col2 = st.columns(2)
            if col1.button("Yes, Delete"):
                # Add code to delete store from database
                st.session_state.confirming_store_delete = None
                st.success("Store deleted successfully!")
                time.sleep(1)
                st.rerun()
            if col2.button("Cancel"):
                st.session_state.confirming_store_delete = None
                st.rerun()
        
        # Display store data (mock implementation)
        if not st.session_state.adding_new_store and not st.session_state.stores_editing and not st.session_state.confirming_store_delete:
            # This would typically fetch from database with pagination and filtering
            # For now, mock data for display
            stores_data = {
                'store_id': ['STR049', 'STR050', 'STR051', 'STR052'],
                'store_establishment_year': [1999, 2005, 2010, 2015],
                'store_size': ['Medium', 'Large', 'Small', 'Medium'],
                'store_location_type': ['Region 1', 'Region 2', 'Region 1', 'Region 3'],
                'store_type': ['Mall Type1', 'Standalone', 'Mall Type2', 'Mall Type1']
            }
            stores_df = pd.DataFrame(stores_data)
            total_stores = len(stores_df)
            
            # Display total count and current range
            total_pages = (total_stores - 1) // st.session_state.stores_per_page + 1
            start_store = st.session_state.stores_page_num * st.session_state.stores_per_page + 1
            end_store = min(start_store + st.session_state.stores_per_page - 1, total_stores)
            
            st.markdown(f"Showing {start_store} to {end_store} of {total_stores} stores")
            
            # Display the stores table
            if not stores_df.empty:
                st.dataframe(
                    stores_df,
                    column_config={
                        "store_id": "Store ID",
                        "store_establishment_year": "Establishment Year",
                        "store_size": "Size",
                        "store_location_type": "Location",
                        "store_type": "Store Type",
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Add pagination controls
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.session_state.stores_page_num > 0:
                        if st.button("← Previous Page"):
                            st.session_state.stores_page_num -= 1
                            st.rerun()
                with col2:
                    st.markdown(f"<div class='center-align'>Page {st.session_state.stores_page_num + 1} of {total_pages}</div>", unsafe_allow_html=True)
                with col3:
                    if st.session_state.stores_page_num < total_pages - 1:
                        if st.button("Next Page →"):
                            st.session_state.stores_page_num += 1
                            st.rerun()
                
                # Add action buttons for each row
                st.markdown("### Actions")
                selected_store = st.selectbox("Select a store to edit or delete:", options=stores_df['store_id'].tolist())
                
                col1, col2 = st.columns(2)
                if col1.button("Edit Selected Store"):
                    # This would typically fetch the store data from database
                    # For now, use the mock data
                    store_data = stores_df[stores_df['store_id'] == selected_store].iloc[0].to_dict()
                    st.session_state.edit_store = store_data
                    st.session_state.stores_editing = True
                    st.rerun()
                
                if col2.button("Delete Selected Store"):
                    st.session_state.confirming_store_delete = selected_store
                    st.rerun()
            else:
                st.info("No stores found matching your criteria.")

# ---- STORES VISUALIZATION PAGE ----
def display_stores_visualization_page():
    st.title("Stores Visualization")
    
    # Mock data for stores
    stores_data = {
        'store_id': ['STR049', 'STR050', 'STR051', 'STR052', 'STR053', 'STR054', 'STR055', 'STR056', 'STR057', 'STR058'],
        'store_establishment_year': [1999, 2005, 2010, 2015, 2000, 2008, 2012, 2018, 2020, 2002],
        'store_size': ['Medium', 'Large', 'Small', 'Medium', 'Large', 'Small', 'Medium', 'Large', 'Small', 'Medium'],
        'store_location_type': ['Region 1', 'Region 2', 'Region 1', 'Region 3', 'Region 2', 'Region 3', 'Region 1', 'Region 2', 'Region 3', 'Region 1'],
        'store_type': ['Mall Type1', 'Standalone', 'Mall Type2', 'Mall Type1', 'Mall Type2', 'Standalone', 'Mall Type1', 'Mall Type2', 'Standalone', 'Mall Type1']
    }
    stores_df = pd.DataFrame(stores_data)
    
    # Add a decade column for visualization
    stores_df['decade'] = (stores_df['store_establishment_year'] // 10) * 10
    stores_df['decade'] = stores_df['decade'].astype(str) + 's'
    
    # Visualization options
    viz_type = st.selectbox(
        "Select Visualization Type",
        options=["Store Count by Type", "Store Count by Size", "Store Count by Location", "Establishment Timeline", "Stores by Decade"]
    )
    
    if viz_type == "Store Count by Type":
        # Count stores by type
        type_counts = stores_df['store_type'].value_counts().reset_index()
        type_counts.columns = ['Store Type', 'Count']
        
        # Create bar chart
        fig = px.bar(
            type_counts,
            x='Store Type',
            y='Count',
            color='Store Type',
            title="Store Count by Store Type"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.markdown("### Data Table")
        st.dataframe(type_counts, hide_index=True, use_container_width=True)
        
    elif viz_type == "Store Count by Size":
        # Count stores by size
        size_counts = stores_df['store_size'].value_counts().reset_index()
        size_counts.columns = ['Store Size', 'Count']
        
        # Create pie chart
        fig = px.pie(
            size_counts,
            values='Count',
            names='Store Size',
            title="Store Distribution by Size"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.markdown("### Data Table")
        st.dataframe(size_counts, hide_index=True, use_container_width=True)
        
    elif viz_type == "Store Count by Location":
        # Count stores by location
        location_counts = stores_df['store_location_type'].value_counts().reset_index()
        location_counts.columns = ['Location', 'Count']
        
        # Create bar chart
        fig = px.bar(
            location_counts,
            x='Location',
            y='Count',
            color='Location',
            title="Store Count by Location"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.markdown("### Data Table")
        st.dataframe(location_counts, hide_index=True, use_container_width=True)
        
    elif viz_type == "Establishment Timeline":
        # Sort by establishment year
        timeline_df = stores_df.sort_values('store_establishment_year')
        
        # Create line chart showing cumulative store count over time
        years = sorted(timeline_df['store_establishment_year'].unique())
        cumulative_counts = []
        count = 0
        
        for year in years:
            count += len(timeline_df[timeline_df['store_establishment_year'] == year])
            cumulative_counts.append(count)
        
        timeline_data = pd.DataFrame({
            'Year': years,
            'Cumulative Store Count': cumulative_counts
        })
        
        fig = px.line(
            timeline_data,
            x='Year',
            y='Cumulative Store Count',
            markers=True,
            title="Cumulative Store Growth Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.markdown("### Data Table")
        st.dataframe(timeline_data, hide_index=True, use_container_width=True)
        
    elif viz_type == "Stores by Decade":
        # Count stores by decade of establishment
        decade_counts = stores_df['decade'].value_counts().reset_index()
        decade_counts.columns = ['Decade', 'Count']
        decade_counts = decade_counts.sort_values('Decade')
        
        # Create bar chart
        fig = px.bar(
            decade_counts,
            x='Decade',
            y='Count',
            color='Decade',
            title="Store Openings by Decade"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.markdown("### Data Table")
        st.dataframe(decade_counts, hide_index=True, use_container_width=True)
    
    # Add insights section
    st.markdown("### Insights")
    
    if viz_type == "Store Count by Type":
        st.write("""
        This visualization shows the distribution of stores by their type. It helps identify which 
        store types are most common in the company's portfolio. The predominant store type can indicate 
        the company's strategic focus in terms of retail presence.
        """)
    elif viz_type == "Store Count by Size":
        st.write("""
        The pie chart displays the proportion of stores by size category (Small, Medium, Large). 
        This distribution can provide insights into the company's real estate strategy and target market.
        Larger stores typically offer more product variety but require higher investment and maintenance costs.
        """)
    elif viz_type == "Store Count by Location":
        st.write("""
        This chart shows how stores are distributed across different regions. Geographic distribution 
        can reveal market penetration strategies and potential areas for expansion. Regions with fewer 
        stores might represent growth opportunities, while regions with many stores may indicate market saturation.
        """)
    elif viz_type == "Establishment Timeline":
        st.write("""
        The timeline visualization shows the cumulative growth of stores over time. Steeper sections 
        of the curve indicate periods of rapid expansion, while flatter sections show slower growth.
        This can be correlated with business strategy changes, economic conditions, or market opportunities.
        """)
    elif viz_type == "Stores by Decade":
        st.write("""
        This chart groups store openings by decade, showing the company's expansion patterns over longer 
        time periods. It helps identify which decades saw the most aggressive expansion and which were 
        more conservative, potentially reflecting broader economic trends or changes in company strategy.
        """)

# ---- STORES STATISTICS PAGE ----
def display_stores_statistics_page():
    st.title("Stores Statistics")
    
    # Mock data for stores
    stores_data = {
        'store_id': ['STR049', 'STR050', 'STR051', 'STR052', 'STR053', 'STR054', 'STR055', 'STR056', 'STR057', 'STR058'],
        'store_establishment_year': [1999, 2005, 2010, 2015, 2000, 2008, 2012, 2018, 2020, 2002],
        'store_size': ['Medium', 'Large', 'Small', 'Medium', 'Large', 'Small', 'Medium', 'Large', 'Small', 'Medium'],
        'store_location_type': ['Region 1', 'Region 2', 'Region 1', 'Region 3', 'Region 2', 'Region 3', 'Region 1', 'Region 2', 'Region 3', 'Region 1'],
        'store_type': ['Mall Type1', 'Standalone', 'Mall Type2', 'Mall Type1', 'Mall Type2', 'Standalone', 'Mall Type1', 'Mall Type2', 'Standalone', 'Mall Type1']
    }
    stores_df = pd.DataFrame(stores_data)
    
    # Create tabs for different types of statistics
    tab1, tab2 = st.tabs(["Summary Statistics", "Categorical Analysis"])
    
    with tab1:
        st.markdown("### Store Establishment Year Statistics")
        
        # Calculate numeric statistics for establishment year
        year_stats = stores_df['store_establishment_year'].describe().to_dict()
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Oldest Store", f"{int(year_stats['min'])}")
            st.metric("Newest Store", f"{int(year_stats['max'])}")
        
        with col2:
            st.metric("Average Age", f"{2025 - int(year_stats['mean']):.1f} years")
            st.metric("Median Year", f"{int(year_stats['50%'])}")
        
        with col3:
            st.metric("Total Stores", f"{int(year_stats['count'])}")
            st.metric("Store Age Range", f"{2025 - int(year_stats['min'])} years")
        
        # Add a histogram for establishment years
        est_year_hist = px.histogram(
            stores_df,
            x='store_establishment_year',
            nbins=10,
            title="Distribution of Store Establishment Years",
            labels={'store_establishment_year': 'Establishment Year', 'count': 'Number of Stores'}
        )
        st.plotly_chart(est_year_hist, use_container_width=True)
        
        # Add a box plot for establishment years
        est_year_box = px.box(
            stores_df,
            y='store_establishment_year',
            title="Box Plot of Store Establishment Years",
            labels={'store_establishment_year': 'Establishment Year'}
        )
        st.plotly_chart(est_year_box, use_container_width=True)
        
        # Store age analysis
        st.markdown("### Store Age Analysis")
        
        # Calculate store ages
        current_year = 2025  # Using the current year from context
        stores_df['store_age'] = current_year - stores_df['store_establishment_year']
        
        # Age categories
        stores_df['age_category'] = pd.cut(
            stores_df['store_age'],
            bins=[0, 5, 10, 15, 20, 25, 30],
            labels=['0-5 years', '6-10 years', '11-15 years', '16-20 years', '21-25 years', '26-30 years']
        )
        
        # Count stores by age category
        age_cat_counts = stores_df['age_category'].value_counts().reset_index()
        age_cat_counts.columns = ['Age Category', 'Count']
        age_cat_counts = age_cat_counts.sort_values('Age Category')
        
        # Create bar chart for age categories
        age_cat_chart = px.bar(
            age_cat_counts,
            x='Age Category',
            y='Count',
            title="Stores by Age Category",
            color='Age Category'
        )
        st.plotly_chart(age_cat_chart, use_container_width=True)
    
    with tab2:
        st.markdown("### Store Attributes Analysis")
        
        # Store size distribution
        st.markdown("#### Store Size Distribution")
        size_counts = stores_df['store_size'].value_counts().reset_index()
        size_counts.columns = ['Store Size', 'Count']
        size_counts['Percentage'] = (size_counts['Count'] / size_counts['Count'].sum() * 100).round(1)
        
        # Display as table
        st.dataframe(
            size_counts,
            column_config={
                "Store Size": st.column_config.TextColumn("Store Size"),
                "Count": st.column_config.NumberColumn("Count", format="%d"),
                "Percentage": st.column_config.NumberColumn("Percentage", format="%.1f%%"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Create pie chart for store sizes
        size_pie = px.pie(
            size_counts,
            values='Count',
            names='Store Size',
            title="Store Size Distribution"
        )
        st.plotly_chart(size_pie, use_container_width=True)
        
        # Store type distribution
        st.markdown("#### Store Type Distribution")
        type_counts = stores_df['store_type'].value_counts().reset_index()
        type_counts.columns = ['Store Type', 'Count']
        type_counts['Percentage'] = (type_counts['Count'] / type_counts['Count'].sum() * 100).round(1)
        
        # Display as table
        st.dataframe(
            type_counts,
            column_config={
                "Store Type": st.column_config.TextColumn("Store Type"),
                "Count": st.column_config.NumberColumn("Count", format="%d"),
                "Percentage": st.column_config.NumberColumn("Percentage", format="%.1f%%"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Create pie chart for store types
        type_pie = px.pie(
            type_counts,
            values='Count',
            names='Store Type',
            title="Store Type Distribution"
        )
        st.plotly_chart(type_pie, use_container_width=True)
        
        # Location distribution
        st.markdown("#### Store Location Distribution")
        location_counts = stores_df['store_location_type'].value_counts().reset_index()
        location_counts.columns = ['Location', 'Count']
        location_counts['Percentage'] = (location_counts['Count'] / location_counts['Count'].sum() * 100).round(1)
        
        # Display as table
        st.dataframe(
            location_counts,
            column_config={
                "Location": st.column_config.TextColumn("Location"),
                "Count": st.column_config.NumberColumn("Count", format="%d"),
                "Percentage": st.column_config.NumberColumn("Percentage", format="%.1f%%"),
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Create pie chart for store locations
        location_pie = px.pie(
            location_counts,
            values='Count',
            names='Location',
            title="Store Location Distribution"
        )
        st.plotly_chart(location_pie, use_container_width=True)
        
        # Cross-tabulation analysis
        st.markdown("#### Cross-Tabulation Analysis")
        
        # Store type by size
        st.markdown("##### Store Type by Size")
        type_by_size = pd.crosstab(stores_df['store_type'], stores_df['store_size'])
        st.dataframe(type_by_size, use_container_width=True)
        
        # Create a grouped bar chart
        type_size_data = []
        for type_name in stores_df['store_type'].unique():
            for size in stores_df['store_size'].unique():
                count = len(stores_df[(stores_df['store_type'] == type_name) & (stores_df['store_size'] == size)])
                type_size_data.append({'Store Type': type_name, 'Store Size': size, 'Count': count})
        
        type_size_df = pd.DataFrame(type_size_data)
        type_size_chart = px.bar(
            type_size_df,
            x='Store Type',
            y='Count',
            color='Store Size',
            barmode='group',
            title="Store Types by Size"
        )
        st.plotly_chart(type_size_chart, use_container_width=True)
        
        # Store type by location
        st.markdown("##### Store Type by Location")
        type_by_location = pd.crosstab(stores_df['store_type'], stores_df['store_location_type'])
        st.dataframe(type_by_location, use_container_width=True)
        
        # Create a grouped bar chart
        type_location_data = []
        for type_name in stores_df['store_type'].unique():
            for location in stores_df['store_location_type'].unique():
                count = len(stores_df[(stores_df['store_type'] == type_name) & (stores_df['store_location_type'] == location)])
                type_location_data.append({'Store Type': type_name, 'Location': location, 'Count': count})
        
        type_location_df = pd.DataFrame(type_location_data)
        type_location_chart = px.bar(
            type_location_df,
            x='Store Type',
            y='Count',
            color='Location',
            barmode='group',
            title="Store Types by Location"
        )
        st.plotly_chart(type_location_chart, use_container_width=True)

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
            
# ---- ABOUT PAGE ----
def display_about_page():
    st.title("About the Textile Industry Management System")
    
    st.markdown("""
    ### Overview
    
    This Streamlit application provides a comprehensive system for managing inventory items with a focus on 
    clothing products. The system allows users to view, add, edit, and delete items, as well as perform data 
    analysis and visualization.
    
    ### Features
    
    1. **Items Management**
       - View all items with pagination
       - Add new items
       - Edit existing items
       - Delete items
       - Filter items based on various criteria
    
    2. **Data Visualization**
       - Multiple chart types including bar charts, pie charts, scatter plots, and more
       - Customizable visualizations
       - Interactive charts for better data exploration
    
    3. **Summary Statistics**
       - Detailed statistics for numeric data
       - Distribution analysis for categorical data
       - Correlation analysis between variables
    
    ### Technical Details
    
    This application is built using:
    - **Streamlit**: For the web interface
    - **PostgreSQL**: For data storage and retrieval
    - **Pandas**: For data manipulation
    - **Plotly**: For interactive visualizations
    - **SciPy**: For statistical analysis
    
    ### Deployment Instructions
    
    #### Prerequisites
    - Python 3.8+
    - PostgreSQL database
    
    #### Installation
    
    1. Clone the repository or download the code
    2. Create a virtual environment:
       ```bash
       python -m venv venv
       source venv/bin/activate  # On Windows: venv\\Scripts\\activate
       ```
    3. Install the required packages:
       ```bash
       pip install streamlit pandas psycopg2-binary plotly scipy streamlit-option-menu
       ```
    4. Configure your PostgreSQL database and update the connection details in the code
    
    #### Running the Application
    
    ```bash
    streamlit run app.py
    ```
    
    The application will be available at http://localhost:8501
    
    ### Database Setup
    
    The application will automatically create the necessary tables if they don't exist. For a production deployment,
    you should set up proper database credentials and security measures.
    
    ### Support
    
    For issues or feature requests, please contact the development team.
    """)

# Run the app
if __name__ == "__main__":
    main()