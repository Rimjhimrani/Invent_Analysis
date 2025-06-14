import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import io
import base64
from matplotlib.figure import Figure
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Enhanced 3D Inventory Analysis with Vendor Filter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .status-excess {
        border-left-color: #007bff !important;
    }
    .status-short {
        border-left-color: #dc3545 !important;
    }
    .status-normal {
        border-left-color: #28a745 !important;
    }
    .vendor-filter {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class InventoryAnalyzer:
    def __init__(self):
        self.status_colors = {
            'Within Norms': '#28a745',      # Green
            'Excess Inventory': '#007bff',   # Blue
            'Short Inventory': '#dc3545'     # Red
        }
        
    def safe_float_convert(self, value):
        """Safely convert string to float, handling commas and other formatting"""
        if pd.isna(value) or value == '' or value is None:
            return 0.0
        
        str_value = str(value).strip()
        str_value = str_value.replace(',', '').replace(' ', '')
        
        if str_value.endswith('%'):
            str_value = str_value[:-1]
        
        try:
            return float(str_value)
        except (ValueError, TypeError):
            return 0.0
    
    def safe_int_convert(self, value):
        """Safely convert string to int, handling commas and other formatting"""
        if pd.isna(value) or value == '' or value is None:
            return 0
        
        str_value = str(value).strip()
        str_value = str_value.replace(',', '').replace(' ', '')
        
        try:
            return int(float(str_value))
        except (ValueError, TypeError):
            return 0
    
    def load_sample_data(self):
        """Load sample inventory data with vendor information"""
        inventory_sample = [
            ["AC0303020106", "FLAT ALUMINIUM PROFILE", "5.230", "4.000", "496", "Vendor_A"],
            ["AC0303020105", "RAIN GUTTER PROFILE", "8.360", "6.000", "1984", "Vendor_B"],
            ["AA0106010001", "HYDRAULIC POWER STEERING OIL", "12.500", "10.000", "2356", "Vendor_A"],
            ["AC0203020077", "Bulb beading LV battery flap", "3.500", "3.000", "248", "Vendor_C"],
            ["AC0303020104", "L- PROFILE JAM PILLAR", "15.940", "20.000", "992", "Vendor_A"],
            ["AA0112014000", "Conduit Pipe Filter to Compressor", "25", "30", "1248", "Vendor_B"],
            ["AA0115120001", "HVPDU ms", "18", "12", "1888", "Vendor_D"],
            ["AA0119020017", "REAR TURN INDICATOR", "35", "40", "1512", "Vendor_C"],
            ["AA0119020019", "REVERSING LAMP", "28", "20", "1152", "Vendor_A"],
            ["AA0822010800", "SIDE DISPLAY BOARD", "42", "50", "2496", "Vendor_B"],
            ["BB0101010001", "ENGINE OIL FILTER", "65", "45", "1300", "Vendor_E"],
            ["BB0202020002", "BRAKE PAD SET", "22", "25", "880", "Vendor_C"],
            ["CC0303030003", "CLUTCH DISC", "8", "12", "640", "Vendor_D"],
            ["DD0404040004", "SPARK PLUG", "45", "35", "450", "Vendor_A"],
            ["EE0505050005", "AIR FILTER", "30", "28", "600", "Vendor_B"],
            ["FF0606060006", "FUEL FILTER", "55", "50", "1100", "Vendor_E"],
            ["GG0707070007", "TRANSMISSION OIL", "40", "35", "800", "Vendor_C"],
            ["HH0808080008", "COOLANT", "22", "30", "660", "Vendor_D"],
            ["II0909090009", "BRAKE FLUID", "15", "12", "300", "Vendor_A"],
            ["JJ1010101010", "WINDSHIELD WASHER", "33", "25", "495", "Vendor_B"]
        ]
        
        inventory_data = []
        for row in inventory_sample:
            inventory_data.append({
                'Material': row[0],
                'Description': row[1],
                'QTY': self.safe_float_convert(row[2]),
                'RM IN QTY': self.safe_float_convert(row[3]),
                'Stock_Value': self.safe_int_convert(row[4]),
                'Vendor': row[5]
            })
        
        return inventory_data
    
    def standardize_inventory_data(self, df):
        """Standardize inventory data and extract QTY, RM, and Vendor columns"""
        if df is None or df.empty:
            return []
        
        # Find required columns (case insensitive)
        qty_columns = ['qty', 'quantity', 'current_qty', 'stock_qty']
        rm_columns = ['rm', 'rm_qty', 'required_qty', 'norm_qty', 'target_qty', 'rm_in_qty', 'ri_in_qty']
        material_columns = ['material', 'material_code', 'part_number', 'item_code', 'code', 'part_no']
        desc_columns = ['description', 'item_description', 'part_description', 'desc']
        value_columns = ['stock_value', 'value', 'amount', 'cost']
        vendor_columns = ['vendor', 'vendor_name', 'supplier', 'supplier_name']
        
        # Get column names (case insensitive)
        available_columns = {k.lower().replace(' ', '_'): k for k in df.columns}
        
        # Find the best matching columns
        qty_col = None
        rm_col = None
        material_col = None
        desc_col = None
        value_col = None
        vendor_col = None
        
        for col_name in qty_columns:
            if col_name in available_columns:
                qty_col = available_columns[col_name]
                break
        
        for col_name in rm_columns:
            if col_name in available_columns:
                rm_col = available_columns[col_name]
                break
        
        for col_name in material_columns:
            if col_name in available_columns:
                material_col = available_columns[col_name]
                break
        
        for col_name in desc_columns:
            if col_name in available_columns:
                desc_col = available_columns[col_name]
                break
        
        for col_name in value_columns:
            if col_name in available_columns:
                value_col = available_columns[col_name]
                break
        
        for col_name in vendor_columns:
            if col_name in available_columns:
                vendor_col = available_columns[col_name]
                break
        
        if not qty_col:
            st.error("QTY/Quantity column not found in inventory file")
            return []
        
        if not rm_col:
            st.error("RM/RM IN QTY column not found in inventory file")
            return []
        
        if not material_col:
            st.error("Material/Part Number column not found in inventory file")
            return []
        
        # Process each record
        standardized_data = []
        for _, record in df.iterrows():
            try:
                material = str(record.get(material_col, '')).strip()
                qty = self.safe_float_convert(record.get(qty_col, 0))
                rm = self.safe_float_convert(record.get(rm_col, 0))
                vendor = str(record.get(vendor_col, 'Unknown')).strip() if vendor_col else 'Unknown'
                
                if material and material.lower() != 'nan' and qty >= 0 and rm >= 0:
                    item = {
                        'Material': material,
                        'Description': str(record.get(desc_col, '')).strip() if desc_col else '',
                        'QTY': qty,
                        'RM IN QTY': rm,
                        'Stock_Value': self.safe_int_convert(record.get(value_col, 0)) if value_col else 0,
                        'Vendor': vendor
                    }
                    standardized_data.append(item)
                    
            except Exception as e:
                continue
        
        return standardized_data
    
    def calculate_variance(self, qty, rm):
        """Calculate variance percentage and absolute value"""
        if rm == 0:
            return 0, 0
        
        variance_percent = ((qty - rm) / rm) * 100
        variance_value = qty - rm
        return variance_percent, variance_value
    
    def determine_status(self, variance_percent, tolerance):
        """Determine inventory status based on variance and tolerance"""
        if abs(variance_percent) <= tolerance:
            return 'Within Norms'
        elif variance_percent > tolerance:
            return 'Excess Inventory'
        else:
            return 'Short Inventory'
    
    def process_data(self, inventory_data, tolerance):
        """Process inventory data and calculate analysis"""
        processed_data = []
        summary_data = {
            'Within Norms': {'count': 0, 'value': 0},
            'Excess Inventory': {'count': 0, 'value': 0},
            'Short Inventory': {'count': 0, 'value': 0}
        }
        
        for item in inventory_data:
            qty = item['QTY']
            rm = item['RM IN QTY']
            stock_value = item['Stock_Value']
            vendor = item['Vendor']
            
            # Calculate variance
            variance_percent, variance_value = self.calculate_variance(qty, rm)
            
            # Determine status
            status = self.determine_status(variance_percent, tolerance)
            
            # Store processed data
            processed_item = {
                'Material': item['Material'],
                'Description': item['Description'],
                'QTY': qty,
                'RM IN QTY': rm,
                'Variance_%': variance_percent,
                'Variance_Value': variance_value,
                'Status': status,
                'Stock_Value': stock_value,
                'Vendor': vendor
            }
            processed_data.append(processed_item)
            
            # Update summary
            summary_data[status]['count'] += 1
            summary_data[status]['value'] += stock_value
        
        return processed_data, summary_data
    
    def get_vendor_summary(self, processed_data):
        """Get summary data by vendor"""
        vendor_summary = {}
        
        for item in processed_data:
            vendor = item['Vendor']
            if vendor not in vendor_summary:
                vendor_summary[vendor] = {
                    'total_parts': 0,
                    'total_qty': 0,
                    'total_rm': 0,
                    'total_value': 0,
                    'short_parts': 0,
                    'excess_parts': 0,
                    'normal_parts': 0
                }
            
            vendor_summary[vendor]['total_parts'] += 1
            vendor_summary[vendor]['total_qty'] += item['QTY']
            vendor_summary[vendor]['total_rm'] += item['RM IN QTY']
            vendor_summary[vendor]['total_value'] += item['Stock_Value']
            
            if item['Status'] == 'Short Inventory':
                vendor_summary[vendor]['short_parts'] += 1
            elif item['Status'] == 'Excess Inventory':
                vendor_summary[vendor]['excess_parts'] += 1
            else:
                vendor_summary[vendor]['normal_parts'] += 1
        
        return vendor_summary

def main():
    # Initialize analyzer
    analyzer = InventoryAnalyzer()
    
    # Header
    st.title("📊 Enhanced 3D Inventory Analysis with Vendor Filter")
    
    st.markdown(
        "<p style='font-size:18px; font-style:italic; margin-top:-10px; text-align:left;'>"
        "Designed and Developed by Agilomatrix - Now with Interactive 3D Visualizations</p>",
        unsafe_allow_html=True
    )
    
    # Sidebar for controls
    st.sidebar.header("⚙️ Control Panel")
    
    # Tolerance setting
    tolerance = st.sidebar.selectbox(
        "Tolerance Zone (+/-)",
        options=[10, 20, 30, 40, 50],
        index=2,  # Default to 30%
        format_func=lambda x: f"{x}%"
    )
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Inventory File",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with QTY, RM IN QTY, and Vendor columns"
    )
    
    # Load data
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            inventory_data = analyzer.standardize_inventory_data(df)
            
            if inventory_data:
                st.sidebar.success(f"✅ Loaded {len(inventory_data)} inventory items")
            else:
                st.sidebar.error("❌ No valid data found in uploaded file")
                inventory_data = analyzer.load_sample_data()
                st.sidebar.info("Using sample data instead")
        
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            inventory_data = analyzer.load_sample_data()
            st.sidebar.info("Using sample data instead")
    else:
        inventory_data = analyzer.load_sample_data()
        st.sidebar.info("📋 Using sample data for demonstration")
    
    # Process data
    processed_data, summary_data = analyzer.process_data(inventory_data, tolerance)
    
    # Get vendor list for filtering
    vendors = sorted(list(set(item['Vendor'] for item in processed_data)))
    
    # Vendor filter
    st.sidebar.header("🏢 Vendor Filter")
    selected_vendor = st.sidebar.selectbox(
        "Select Vendor (for Short Inventory focus)",
        options=['All Vendors'] + vendors,
        help="Select a specific vendor to focus on their short inventory items"
    )
    
    # Apply vendor filter for short inventory focus
    if selected_vendor != 'All Vendors':
        st.markdown(f'<div class="vendor-filter">🏢 <strong>Vendor Focus:</strong> {selected_vendor} - Showing Short Inventory Analysis</div>', unsafe_allow_html=True)
        
        # Filter data for selected vendor and short inventory
        vendor_short_items = [item for item in processed_data if item['Vendor'] == selected_vendor and item['Status'] == 'Short Inventory']
        
        if vendor_short_items:
            st.info(f"Found {len(vendor_short_items)} short inventory items for {selected_vendor}")
        else:
            st.success(f"No short inventory items found for {selected_vendor}")
    
    # Display status criteria
    st.info(f"""
    **Status Criteria (Tolerance: ±{tolerance}%)**
    - 🟢 **Within Norms**: QTY = RM IN QTY ± {tolerance}%
    - 🔵 **Excess Inventory**: QTY > RM IN QTY + {tolerance}%
    - 🔴 **Short Inventory**: QTY < RM IN QTY - {tolerance}%
    """)
    
    # Summary Dashboard
    st.header("📈 Summary Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card status-normal">', unsafe_allow_html=True)
        st.metric(
            label="🟢 Within Norms",
            value=f"{summary_data['Within Norms']['count']} parts",
            delta=f"₹{summary_data['Within Norms']['value']:,}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card status-excess">', unsafe_allow_html=True)
        st.metric(
            label="🔵 Excess Inventory",
            value=f"{summary_data['Excess Inventory']['count']} parts",
            delta=f"₹{summary_data['Excess Inventory']['value']:,}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card status-short">', unsafe_allow_html=True)
        st.metric(
            label="🔴 Short Inventory",
            value=f"{summary_data['Short Inventory']['count']} parts",
            delta=f"₹{summary_data['Short Inventory']['value']:,}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Vendor Summary
    vendor_summary = analyzer.get_vendor_summary(processed_data)
    
    st.header("🏢 Vendor Summary")
    vendor_df = pd.DataFrame([
        {
            'Vendor': vendor,
            'Total Parts': data['total_parts'],
            'Total QTY': round(data['total_qty'], 2),
            'Total RM': round(data['total_rm'], 2),
            'Short Items': data['short_parts'],
            'Excess Items': data['excess_parts'],
            'Normal Items': data['normal_parts'],
            'Total Value': f"₹{data['total_value']:,}"
        }
        for vendor, data in vendor_summary.items()
    ])
    
    st.dataframe(vendor_df, use_container_width=True, hide_index=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 3D Interactive Graphs", "📊 2D Analysis", "📋 Detailed Data", "🏢 Vendor Analysis", "📤 Export"])
    
    with tab1:
        st.header("🎯 3D Interactive Visualizations")
        st.info("🖱️ **Tip:** Hover over points to see detailed information including part numbers and descriptions. Use mouse to rotate and zoom the 3D plots!")
        
        # 3D Scatter Plot - QTY vs RM vs Stock Value
        st.subheader("📊 3D Scatter: QTY vs RM vs Stock Value")
        
        # Prepare data for 3D scatter
        df_3d = pd.DataFrame(processed_data)
        
        # Create hover text with all details
        hover_text = []
        for _, row in df_3d.iterrows():
            hover_info = (
                f"<b>Part No:</b> {row['Material']}<br>"
                f"<b>Description:</b> {row['Description']}<br>"
                f"<b>Vendor:</b> {row['Vendor']}<br>"
                f"<b>Current QTY:</b> {row['QTY']}<br>"
                f"<b>RM QTY:</b> {row['RM IN QTY']}<br>"
                f"<b>Stock Value:</b> ₹{row['Stock_Value']:,}<br>"
                f"<b>Variance:</b> {row['Variance_%']:.1f}%<br>"
                f"<b>Status:</b> {row['Status']}"
            )
            hover_text.append(hover_info)
        
        df_3d['hover_text'] = hover_text
        
        fig_3d_scatter = px.scatter_3d(
            df_3d, 
            x='QTY', 
            y='RM IN QTY', 
            z='Stock_Value',
            color='Status',
            color_discrete_map=analyzer.status_colors,
            title="3D Inventory Analysis: QTY vs RM vs Stock Value",
            labels={
                'QTY': 'Current Quantity',
                'RM IN QTY': 'Required Quantity',
                'Stock_Value': 'Stock Value (₹)'
            },
            hover_data={'hover_text': True},
            size_max=15
        )
        
        # Update hover template
        fig_3d_scatter.update_traces(
            hovertemplate='%{customdata[0]}<extra></extra>',
            customdata=df_3d[['hover_text']].values
        )
        
        fig_3d_scatter.update_layout(
            scene=dict(
                xaxis_title='Current Quantity',
                yaxis_title='Required Quantity',
                zaxis_title='Stock Value (₹)',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            height=600
        )
        
            st.plotly_chart(fig_3d_surface, use_container_width=True)
        else:
            st.info("Need at least 3 vendors for 3D surface plot. Showing vendor scatter plot instead.")
            
            # Alternative: 3D scatter plot for vendors
            fig_vendor_scatter = px.scatter_3d(
                pd.DataFrame(vendor_perf_data),
                x='short_pct',
                y='excess_pct',
                z='avg_value',
                color='vendor',
                size='total_parts',
                title="3D Vendor Performance Scatter",
                labels={
                    'short_pct': 'Short Inventory %',
                    'excess_pct': 'Excess Inventory %',
                    'avg_value': 'Average Stock Value (₹)'
                }
            )
            
            st.plotly_chart(fig_vendor_scatter, use_container_width=True)
    
    with tab2:
        st.header("📊 2D Analysis Charts")
        
        # Create 2D visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution pie chart
            status_counts = [summary_data[status]['count'] for status in analyzer.status_colors.keys()]
            status_labels = list(analyzer.status_colors.keys())
            
            fig_pie = px.pie(
                values=status_counts,
                names=status_labels,
                title="Inventory Status Distribution",
                color=status_labels,
                color_discrete_map=analyzer.status_colors
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Variance distribution histogram
            variances = [item['Variance_%'] for item in processed_data]
            fig_hist = px.histogram(
                x=variances,
                nbins=20,
                title="Variance Distribution",
                labels={'x': 'Variance %', 'y': 'Count'}
            )
            fig_hist.add_vline(x=tolerance, line_dash="dash", line_color="green", annotation_text=f"+{tolerance}%")
            fig_hist.add_vline(x=-tolerance, line_dash="dash", line_color="red", annotation_text=f"-{tolerance}%")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # QTY vs RM scatter plot
        df_2d = pd.DataFrame(processed_data)
        fig_scatter = px.scatter(
            df_2d,
            x='RM IN QTY',
            y='QTY',
            color='Status',
            size='Stock_Value',
            hover_data=['Material', 'Description', 'Vendor', 'Variance_%'],
            title="Current QTY vs Required QTY",
            color_discrete_map=analyzer.status_colors
        )
        
        # Add diagonal line (perfect match)
        max_val = max(df_2d['QTY'].max(), df_2d['RM IN QTY'].max())
        fig_scatter.add_shape(
            type="line",
            x0=0, y0=0,
            x1=max_val, y1=max_val,
            line=dict(dash="dash", color="gray"),
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Vendor comparison bar chart
        vendor_comparison = pd.DataFrame([
            {
                'Vendor': vendor,
                'Short Items': data['short_parts'],
                'Excess Items': data['excess_parts'],
                'Normal Items': data['normal_parts']
            }
            for vendor, data in vendor_summary.items()
        ])
        
        fig_vendor_bar = px.bar(
            vendor_comparison.melt(id_vars=['Vendor'], var_name='Status', value_name='Count'),
            x='Vendor',
            y='Count',
            color='Status',
            title="Vendor Performance Comparison",
            color_discrete_map={
                'Short Items': '#dc3545',
                'Excess Items': '#007bff',
                'Normal Items': '#28a745'
            }
        )
        st.plotly_chart(fig_vendor_bar, use_container_width=True)
    
    with tab3:
        st.header("📋 Detailed Inventory Data")
        
        # Display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.selectbox(
                "Filter by Status",
                options=['All'] + list(analyzer.status_colors.keys())
            )
        
        with col2:
            vendor_filter = st.selectbox(
                "Filter by Vendor",
                options=['All'] + vendors
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                options=['Material', 'QTY', 'RM IN QTY', 'Variance_%', 'Stock_Value'],
                index=3  # Default to Variance_%
            )
        
        # Apply filters
        filtered_data = processed_data.copy()
        
        if status_filter != 'All':
            filtered_data = [item for item in filtered_data if item['Status'] == status_filter]
        
        if vendor_filter != 'All':
            filtered_data = [item for item in filtered_data if item['Vendor'] == vendor_filter]
        
        # Sort data
        reverse_sort = sort_by in ['QTY', 'RM IN QTY', 'Stock_Value']
        filtered_data = sorted(filtered_data, key=lambda x: abs(x[sort_by]) if sort_by == 'Variance_%' else x[sort_by], reverse=reverse_sort)
        
        # Convert to DataFrame for display
        display_df = pd.DataFrame(filtered_data)
        if not display_df.empty:
            display_df['Variance_%'] = display_df['Variance_%'].round(2)
            display_df['Stock_Value'] = display_df['Stock_Value'].apply(lambda x: f"₹{x:,}")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Material': st.column_config.TextColumn('Part Number', width='medium'),
                    'Description': st.column_config.TextColumn('Description', width='large'),
                    'QTY': st.column_config.NumberColumn('Current QTY', format='%.2f'),
                    'RM IN QTY': st.column_config.NumberColumn('Required QTY', format='%.2f'),
                    'Variance_%': st.column_config.NumberColumn('Variance %', format='%.2f%%'),
                    'Variance_Value': st.column_config.NumberColumn('Variance Value', format='%.2f'),
                    'Status': st.column_config.TextColumn('Status', width='medium'),
                    'Stock_Value': st.column_config.TextColumn('Stock Value', width='medium'),
                    'Vendor': st.column_config.TextColumn('Vendor', width='medium')
                }
            )
            
            st.info(f"Showing {len(filtered_data)} items out of {len(processed_data)} total")
        else:
            st.warning("No data matches the selected filters")
    
    with tab4:
        st.header("🏢 Detailed Vendor Analysis")
        
        # Vendor selection for detailed analysis
        selected_analysis_vendor = st.selectbox(
            "Select Vendor for Detailed Analysis",
            options=vendors,
            help="Choose a vendor to see detailed breakdown of their inventory"
        )
        
        if selected_analysis_vendor:
            vendor_items = [item for item in processed_data if item['Vendor'] == selected_analysis_vendor]
            vendor_stats = vendor_summary[selected_analysis_vendor]
            
            # Vendor overview metrics
            st.subheader(f"📊 {selected_analysis_vendor} Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Parts", vendor_stats['total_parts'])
            
            with col2:
                st.metric("Total QTY", f"{vendor_stats['total_qty']:.2f}")
            
            with col3:
                st.metric("Required QTY", f"{vendor_stats['total_rm']:.2f}")
            
            with col4:
                st.metric("Total Value", f"₹{vendor_stats['total_value']:,}")
            
            # Status breakdown for selected vendor
            col1, col2 = st.columns(2)
            
            with col1:
                vendor_status_data = {
                    'Status': ['Normal Items', 'Short Items', 'Excess Items'],
                    'Count': [vendor_stats['normal_parts'], vendor_stats['short_parts'], vendor_stats['excess_parts']]
                }
                
                fig_vendor_pie = px.pie(
                    values=vendor_status_data['Count'],
                    names=vendor_status_data['Status'],
                    title=f"{selected_analysis_vendor} Status Distribution",
                    color_discrete_map={
                        'Normal Items': '#28a745',
                        'Short Items': '#dc3545',
                        'Excess Items': '#007bff'
                    }
                )
                st.plotly_chart(fig_vendor_pie, use_container_width=True)
            
            with col2:
                # Top 10 items by variance for selected vendor
                vendor_items_sorted = sorted(vendor_items, key=lambda x: abs(x['Variance_%']), reverse=True)[:10]
                
                if vendor_items_sorted:
                    variance_df = pd.DataFrame(vendor_items_sorted)
                    fig_vendor_variance = px.bar(
                        variance_df,
                        x='Material',
                        y='Variance_%',
                        color='Status',
                        title=f"Top 10 Items by Variance - {selected_analysis_vendor}",
                        color_discrete_map=analyzer.status_colors
                    )
                    fig_vendor_variance.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_vendor_variance, use_container_width=True)
            
            # Detailed vendor items table
            st.subheader(f"📋 All Items for {selected_analysis_vendor}")
            vendor_df = pd.DataFrame(vendor_items)
            vendor_df['Variance_%'] = vendor_df['Variance_%'].round(2)
            vendor_df['Stock_Value'] = vendor_df['Stock_Value'].apply(lambda x: f"₹{x:,}")
            
            st.dataframe(
                vendor_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Material': st.column_config.TextColumn('Part Number'),
                    'Description': st.column_config.TextColumn('Description'),
                    'QTY': st.column_config.NumberColumn('Current QTY', format='%.2f'),
                    'RM IN QTY': st.column_config.NumberColumn('Required QTY', format='%.2f'),
                    'Variance_%': st.column_config.NumberColumn('Variance %', format='%.2f%%'),
                    'Status': st.column_config.TextColumn('Status'),
                    'Stock_Value': st.column_config.TextColumn('Stock Value')
                }
            )
    
    with tab5:
        st.header("📤 Export Options")
        
        # Export full report
        if st.button("📊 Generate Full Analysis Report", type="primary"):
            # Create comprehensive report
            report_data = []
            
            # Summary section
            report_data.append("=== INVENTORY ANALYSIS REPORT ===")
            report_data.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_data.append(f"Tolerance Setting: ±{tolerance}%")
            report_data.append("")
            
            # Overall summary
            report_data.append("=== OVERALL SUMMARY ===")
            total_parts = sum(summary_data[status]['count'] for status in summary_data)
            total_value = sum(summary_data[status]['value'] for status in summary_data)
            
            report_data.append(f"Total Parts Analyzed: {total_parts}")
            report_data.append(f"Total Stock Value: ₹{total_value:,}")
            report_data.append("")
            
            for status in analyzer.status_colors.keys():
                count = summary_data[status]['count']
                value = summary_data[status]['value']
                percentage = (count / total_parts) * 100 if total_parts > 0 else 0
                report_data.append(f"{status}: {count} parts ({percentage:.1f}%) - ₹{value:,}")
            
            report_data.append("")
            
            # Vendor summary
            report_data.append("=== VENDOR SUMMARY ===")
            for vendor, data in vendor_summary.items():
                report_data.append(f"\n{vendor}:")
                report_data.append(f"  Total Parts: {data['total_parts']}")
                report_data.append(f"  Short Items: {data['short_parts']}")
                report_data.append(f"  Excess Items: {data['excess_parts']}")
                report_data.append(f"  Normal Items: {data['normal_parts']}")
                report_data.append(f"  Total Value: ₹{data['total_value']:,}")
            
            report_data.append("")
            
            # Critical items (high variance)
            report_data.append("=== CRITICAL ITEMS (>50% Variance) ===")
            critical_items = [item for item in processed_data if abs(item['Variance_%']) > 50]
            critical_items = sorted(critical_items, key=lambda x: abs(x['Variance_%']), reverse=True)
            
            for item in critical_items[:20]:  # Top 20 critical items
                report_data.append(f"{item['Material']} - {item['Description'][:50]}...")
                report_data.append(f"  Current QTY: {item['QTY']}, Required: {item['RM IN QTY']}")
                report_data.append(f"  Variance: {item['Variance_%']:.1f}% - {item['Status']}")
                report_data.append(f"  Vendor: {item['Vendor']}, Value: ₹{item['Stock_Value']:,}")
                report_data.append("")
            
            # Convert to downloadable format
            report_text = "\n".join(report_data)
            
            st.download_button(
                label="📥 Download Analysis Report",
                data=report_text,
                file_name=f"inventory_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        # Export detailed data
        col1, col2 = st.columns(2)
        
        with col1:
            # Export filtered data as CSV
            export_df = pd.DataFrame(processed_data)
            csv_data = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Detailed Data (CSV)",
                data=csv_data,
                file_name=f"inventory_detailed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export vendor summary
            vendor_summary_df = pd.DataFrame([
                {
                    'Vendor': vendor,
                    'Total_Parts': data['total_parts'],
                    'Short_Items': data['short_parts'],
                    'Excess_Items': data['excess_parts'],
                    'Normal_Items': data['normal_parts'],
                    'Total_QTY': data['total_qty'],
                    'Total_RM': data['total_rm'],
                    'Total_Value': data['total_value']
                }
                for vendor, data in vendor_summary.items()
            ])
            
            vendor_csv = vendor_summary_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Vendor Summary (CSV)",
                data=vendor_csv,
                file_name=f"vendor_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Instructions for further analysis
        st.subheader("📋 Next Steps")
        st.info("""
        **Recommended Actions:**
        1. **Short Inventory Items**: Review procurement plans and expedite orders
        2. **Excess Inventory Items**: Consider reducing future orders or redistribute stock
        3. **High-Variance Items**: Investigate root causes and improve forecasting
        4. **Vendor Performance**: Discuss inventory management strategies with underperforming vendors
        
        **Tips for Better Inventory Management:**
        - Set up automated alerts for items approaching critical thresholds
        - Implement ABC analysis to prioritize high-value items
        - Regular vendor performance reviews
        - Consider implementing Just-In-Time (JIT) inventory practices
        """)

if __name__ == "__main__":
    main()
