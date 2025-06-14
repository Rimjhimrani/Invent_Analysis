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
    page_title="Enhanced Inventory Analysis with Vendor Filter",
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
    .viz-description {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
        font-size: 0.9rem;
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

def create_3d_scatter_plot(processed_data, analyzer):
    """Create 3D scatter plot with QTY, RM, and Stock Value"""
    st.markdown("""
    <div class="viz-description">
    <strong>📊 3D Inventory Scatter Plot</strong><br>
    This 3D visualization shows the relationship between Current Quantity (X-axis), Required Quantity (Y-axis), 
    and Stock Value (Z-axis). Each point represents a material item, colored by its inventory status. 
    Hover over points to see detailed information including material code, vendor, and variance percentage.
    </div>
    """, unsafe_allow_html=True)
    
    df = pd.DataFrame(processed_data)
    
    fig = px.scatter_3d(
        df, 
        x='QTY', 
        y='RM IN QTY', 
        z='Stock_Value',
        color='Status',
        color_discrete_map=analyzer.status_colors,
        hover_data=['Material', 'Vendor', 'Variance_%'],
        title="3D Inventory Analysis: QTY vs RM vs Stock Value",
        labels={
            'QTY': 'Current Quantity',
            'RM IN QTY': 'Required Quantity',
            'Stock_Value': 'Stock Value (₹)'
        }
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Current Quantity',
            yaxis_title='Required Quantity',
            zaxis_title='Stock Value (₹)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600
    )
    
    return fig

def create_3d_surface_plot(processed_data):
    """Create 3D surface plot for variance analysis"""
    st.markdown("""
    <div class="viz-description">
    <strong>🌊 3D Variance Surface Plot</strong><br>
    This surface plot visualizes the variance landscape across different quantity ranges. The surface shows 
    how variance percentage changes with current quantity (X-axis) and required quantity (Y-axis). 
    Higher peaks indicate greater variance, helping identify patterns in inventory deviations.
    </div>
    """, unsafe_allow_html=True)
    
    df = pd.DataFrame(processed_data)
    
    # Create grid for surface plot
    qty_range = np.linspace(df['QTY'].min(), df['QTY'].max(), 20)
    rm_range = np.linspace(df['RM IN QTY'].min(), df['RM IN QTY'].max(), 20)
    
    QTY_grid, RM_grid = np.meshgrid(qty_range, rm_range)
    
    # Calculate variance for each grid point
    variance_grid = np.zeros_like(QTY_grid)
    for i in range(len(qty_range)):
        for j in range(len(rm_range)):
            if rm_range[j] != 0:
                variance_grid[j, i] = ((qty_range[i] - rm_range[j]) / rm_range[j]) * 100
    
    fig = go.Figure(data=[go.Surface(
        z=variance_grid,
        x=QTY_grid,
        y=RM_grid,
        colorscale='RdYlBu',
        showscale=True,
        colorbar=dict(title="Variance %")
    )])
    
    fig.update_layout(
        title="3D Variance Surface Analysis",
        scene=dict(
            xaxis_title='Current Quantity',
            yaxis_title='Required Quantity',
            zaxis_title='Variance %',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600
    )
    
    return fig

def create_3d_vendor_bubble_chart(vendor_summary):
    """Create 3D bubble chart for vendor analysis"""
    st.markdown("""
    <div class="viz-description">
    <strong>🎈 3D Vendor Performance Bubbles</strong><br>
    This 3D bubble chart displays vendor performance across multiple dimensions. X-axis shows total parts, 
    Y-axis shows total quantity, Z-axis shows total value, and bubble size represents the number of short items. 
    Larger bubbles indicate vendors with more shortage issues. Colors represent different vendors.
    </div>
    """, unsafe_allow_html=True)
    
    vendors = list(vendor_summary.keys())
    x_vals = [vendor_summary[v]['total_parts'] for v in vendors]
    y_vals = [vendor_summary[v]['total_qty'] for v in vendors]
    z_vals = [vendor_summary[v]['total_value'] for v in vendors]
    sizes = [vendor_summary[v]['short_parts'] * 10 + 10 for v in vendors]  # Scale for visibility
    
    colors = px.colors.qualitative.Set3[:len(vendors)]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8,
            line=dict(width=2, color='black')
        ),
        text=vendors,
        hovertemplate='<b>%{text}</b><br>' +
                      'Total Parts: %{x}<br>' +
                      'Total QTY: %{y:.2f}<br>' +
                      'Total Value: ₹%{z:,}<br>' +
                      '<extra></extra>'
    )])
    
    fig.update_layout(
        title="3D Vendor Performance Analysis",
        scene=dict(
            xaxis_title='Total Parts',
            yaxis_title='Total Quantity',
            zaxis_title='Total Value (₹)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600
    )
    
    return fig

def create_3d_status_pie_chart(summary_data, analyzer):
    """Create 3D pie chart for status distribution"""
    st.markdown("""
    <div class="viz-description">
    <strong>🥧 3D Status Distribution Pie</strong><br>
    This enhanced 3D pie chart shows the distribution of inventory status across all materials. 
    The three-dimensional effect provides better visual impact while maintaining the clarity of proportions. 
    Each slice represents a different inventory status with corresponding colors and percentages.
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data
    labels = []
    values = []
    colors = []
    
    for status, data in summary_data.items():
        if data['count'] > 0:
            labels.append(status)
            values.append(data['count'])
            colors.append(analyzer.status_colors[status])
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent',
        textposition='outside',
        pull=[0.1 if status == 'Short Inventory' else 0 for status in labels]  # Pull out short inventory slice
    )])
    
    fig.update_layout(
        title="3D Enhanced Status Distribution",
        font=dict(size=14),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_3d_cylinder_chart(processed_data, analyzer):
    """Create 3D cylinder chart for top materials by variance"""
    st.markdown("""
    <div class="viz-description">
    <strong>🏗️ 3D Material Variance Cylinders</strong><br>
    This 3D cylindrical chart displays the top 10 materials with highest absolute variance values. 
    Each cylinder's height represents the variance magnitude, and colors indicate the status type. 
    This visualization helps identify which materials have the most significant inventory deviations.
    </div>
    """, unsafe_allow_html=True)
    
    # Get top 10 materials by absolute variance
    sorted_data = sorted(processed_data, key=lambda x: abs(x['Variance_Value']), reverse=True)[:10]
    
    materials = [item['Material'][:10] + '...' if len(item['Material']) > 10 else item['Material'] for item in sorted_data]
    variances = [abs(item['Variance_Value']) for item in sorted_data]
    colors = [analyzer.status_colors[item['Status']] for item in sorted_data]
    
    # Create 3D bar chart that looks like cylinders
    fig = go.Figure(data=[go.Bar(
        x=materials,
        y=variances,
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.8)', width=1.5),
            opacity=0.9
        ),
        hovertemplate='<b>%{x}</b><br>' +
                      'Variance: %{y:.2f}<br>' +
                      '<extra></extra>'
    )])
    
    fig.update_layout(
        title="3D Top 10 Materials by Variance (Cylinder View)",
        xaxis_title="Material Code",
        yaxis_title="Absolute Variance Value",
        xaxis=dict(tickangle=45),
        height=500,
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='white'
    )
    
    return fig

def create_3d_vendor_performance_radar(vendor_summary):
    """Create 3D-style radar chart for vendor performance"""
    st.markdown("""
    <div class="viz-description">
    <strong>🎯 3D Vendor Performance Radar</strong><br>
    This radar chart provides a comprehensive view of vendor performance across multiple metrics. 
    Each axis represents a different performance indicator: total parts, quantity efficiency, 
    value contribution, and shortage ratio. The area covered by each vendor's line indicates overall performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Select top 5 vendors by total parts for clarity
    top_vendors = sorted(vendor_summary.items(), key=lambda x: x[1]['total_parts'], reverse=True)[:5]
    
    fig = go.Figure()
    
    categories = ['Total Parts', 'Total QTY', 'Total Value', 'Normal Items', 'Performance Score']
    
    for vendor, data in top_vendors:
        # Normalize values for radar chart (0-100 scale)
        max_parts = max(v['total_parts'] for v in vendor_summary.values())
        max_qty = max(v['total_qty'] for v in vendor_summary.values())
        max_value = max(v['total_value'] for v in vendor_summary.values())
        max_normal = max(v['normal_parts'] for v in vendor_summary.values())
        
        performance_score = 100 - (data['short_parts'] / data['total_parts'] * 100) if data['total_parts'] > 0 else 100
        
        values = [
            (data['total_parts'] / max_parts) * 100,
            (data['total_qty'] / max_qty) * 100,
            (data['total_value'] / max_value) * 100,
            (data['normal_parts'] / max_normal) * 100,
            performance_score
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            name=vendor,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title="3D Vendor Performance Radar Analysis",
        height=600,
        showlegend=True
    )
    
    return fig

def main():
    # Initialize analyzer
    analyzer = InventoryAnalyzer()
    
    # Header
    st.title("📊 Inventory Analysis with Vendor Filter")
    
    st.markdown(
        "<p style='font-size:18px; font-style:italic; margin-top:-10px; text-align:left;'>"
        "Designed and Developed by Agilomatrix</p>",
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
    
    st.header("🏢 Vendor Analysis")
    
    # Create vendor summary table
    vendor_df = []
    for vendor, data in vendor_summary.items():
        vendor_df.append({
            'Vendor': vendor,
            'Total Parts': data['total_parts'],
            'Total QTY': f"{data['total_qty']:.2f}",
            'Total Value (₹)': f"{data['total_value']:,}",
            'Short Items': data['short_parts'],
            'Excess Items': data['excess_parts'],
            'Normal Items': data['normal_parts'],
            'Performance %': f"{((data['normal_parts'] / data['total_parts']) * 100):.1f}%" if data['total_parts'] > 0 else "0%"
        })
    
    st.dataframe(pd.DataFrame(vendor_df), use_container_width=True)
    
    # 3D Visualizations Section
    st.header("🎯 3D Advanced Visualizations")
    
    # Create tabs for different 3D visualizations
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "3D Scatter Plot", 
        "3D Surface Plot", 
        "3D Vendor Bubbles", 
        "3D Status Pie", 
        "3D Material Cylinders",
        "3D Vendor Radar"
    ])
    
    with tab1:
        fig_3d_scatter = create_3d_scatter_plot(processed_data, analyzer)
        st.plotly_chart(fig_3d_scatter, use_container_width=True)
    
    with tab2:
        fig_3d_surface = create_3d_surface_plot(processed_data)
        st.plotly_chart(fig_3d_surface, use_container_width=True)
    
    with tab3:
        fig_3d_vendor_bubble = create_3d_vendor_bubble_chart(vendor_summary)
        st.plotly_chart(fig_3d_vendor_bubble, use_container_width=True)
    
    with tab4:
        fig_3d_pie = create_3d_status_pie_chart(summary_data, analyzer)
        st.plotly_chart(fig_3d_pie, use_container_width=True)
    
    with tab5:
        fig_3d_cylinder = create_3d_cylinder_chart(processed_data, analyzer)
        st.plotly_chart(fig_3d_cylinder, use_container_width=True)
    
    with tab6:
        fig_3d_radar = create_3d_vendor_performance_radar(vendor_summary)
        st.plotly_chart(fig_3d_radar, use_container_width=True)
    
    # Detailed Analysis Section
    st.header("📋 Detailed Analysis")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            options=['All'] + list(summary_data.keys())
        )
    
    with col2:
        vendor_filter = st.selectbox(
            "Filter by Vendor",
            options=['All'] + vendors
        )
    
    # Apply filters
    filtered_data = processed_data
    
    if status_filter != 'All':
        filtered_data = [item for item in filtered_data if item['Status'] == status_filter]
    
    if vendor_filter != 'All':
        filtered_data = [item for item in filtered_data if item['Vendor'] == vendor_filter]
    
    # Display filtered data
    if filtered_data:
        st.subheader(f"📊 Filtered Results ({len(filtered_data)} items)")
        
        # Convert to DataFrame for display
        display_df = pd.DataFrame(filtered_data)
        
        # Format numeric columns
        display_df['QTY'] = display_df['QTY'].apply(lambda x: f"{x:.2f}")
        display_df['RM IN QTY'] = display_df['RM IN QTY'].apply(lambda x: f"{x:.2f}")
        display_df['Variance_%'] = display_df['Variance_%'].apply(lambda x: f"{x:.2f}%")
        display_df['Variance_Value'] = display_df['Variance_Value'].apply(lambda x: f"{x:.2f}")
        display_df['Stock_Value'] = display_df['Stock_Value'].apply(lambda x: f"₹{x:,}")
        
        # Style the dataframe
        def highlight_status(row):
            if row['Status'] == 'Short Inventory':
                return ['background-color: #ffebee'] * len(row)
            elif row['Status'] == 'Excess Inventory':
                return ['background-color: #e3f2fd'] * len(row)
            else:
                return ['background-color: #e8f5e8'] * len(row)
        
        styled_df = display_df.style.apply(highlight_status, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Download option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"inventory_analysis_{status_filter}_{vendor_filter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("No data matches the selected filters.")
    
    # Short Inventory Focus (if vendor selected)
    if selected_vendor != 'All Vendors':
        st.header(f"🔍 {selected_vendor} - Short Inventory Focus")
        
        vendor_short_items = [item for item in processed_data if item['Vendor'] == selected_vendor and item['Status'] == 'Short Inventory']
        
        if vendor_short_items:
            # Summary for short items
            total_short_value = sum(item['Stock_Value'] for item in vendor_short_items)
            total_shortage_qty = sum(abs(item['Variance_Value']) for item in vendor_short_items if item['Variance_Value'] < 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="🔴 Short Items Count",
                    value=len(vendor_short_items)
                )
            
            with col2:
                st.metric(
                    label="💰 Short Items Value",
                    value=f"₹{total_short_value:,}"
                )
            
            with col3:
                st.metric(
                    label="📦 Total Shortage Qty",
                    value=f"{total_shortage_qty:.2f}"
                )
            
            # Display short items table
            short_df = pd.DataFrame(vendor_short_items)
            short_df['QTY'] = short_df['QTY'].apply(lambda x: f"{x:.2f}")
            short_df['RM IN QTY'] = short_df['RM IN QTY'].apply(lambda x: f"{x:.2f}")
            short_df['Variance_%'] = short_df['Variance_%'].apply(lambda x: f"{x:.2f}%")
            short_df['Variance_Value'] = short_df['Variance_Value'].apply(lambda x: f"{x:.2f}")
            short_df['Stock_Value'] = short_df['Stock_Value'].apply(lambda x: f"₹{x:,}")
            
            st.dataframe(short_df, use_container_width=True)
            
            # Download short inventory report
            csv_short = short_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {selected_vendor} Short Inventory Report",
                data=csv_short,
                file_name=f"short_inventory_{selected_vendor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        else:
            st.success(f"✅ Great! {selected_vendor} has no short inventory items.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 14px;'>"
        "📊 Enhanced Inventory Analysis Dashboard | "
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "Powered by Streamlit & Plotly"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
