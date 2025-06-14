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
            
    def create_stock_value_chart(self, analysis_data):
        """Create bar chart showing top 10 parts by stock value"""
        # Sort by stock value and get top 10
        sorted_data = sorted(analysis_data, key=lambda x: x['Stock_Value'], reverse=True)[:10]
        
        materials = [item['Material'] for item in sorted_data]
        values = [item['Stock_Value'] for item in sorted_data]
        colors = [self.status_colors[item['Status']] for item in sorted_data]
        
        fig = go.Figure(data=[go.Bar(
            x=materials,
            y=values,
            marker_color=colors,
            text=[f'₹{v:,}' for v in values],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Top 10 Parts by Stock Value",
            xaxis_title="Material Code",
            yaxis_title="Stock Value (₹)",
            height=400,
            xaxis_tickangle=-45,
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            yaxis=dict(
                tickformat=',.0f'
            )
        )
        
        return fig
    
    def create_variance_chart(self, analysis_data):
        """Create bar chart showing top 10 materials by variance"""
        # Sort by absolute variance and get top 10
        sorted_data = sorted(analysis_data, key=lambda x: abs(x['Variance_%']), reverse=True)[:10]
        
        materials = [item['Material'] for item in sorted_data]
        variances = [item['Variance_%'] for item in sorted_data]
        colors = [self.status_colors[item['Status']] for item in sorted_data]
        
        fig = go.Figure(data=[go.Bar(
            x=materials,
            y=variances,
            marker_color=colors,
            text=[f'{v:.1f}%' for v in variances],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Top 10 Materials by Variance",
            xaxis_title="Material Code",
            yaxis_title="Variance %",
            height=400,
            xaxis_tickangle=-45,
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            yaxis=dict(
                tickformat='.1f',
                ticksuffix='%'
            )
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        return fig
    
    def create_comparison_chart(self, analysis_data):
        """Create grouped bar chart comparing QTY vs RM"""
        # Get top 10 by stock value
        sorted_data = sorted(analysis_data, key=lambda x: x['Stock_Value'], reverse=True)[:10]
        
        materials = [item['Material'] for item in sorted_data]
        qty_values = [item['QTY'] for item in sorted_data]
        rm_values = [item['RM IN QTY'] for item in sorted_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current QTY',
            x=materials,
            y=qty_values,
            marker_color='#17a2b8',
            opacity=0.8,
            text=[f'{v:.1f}' for v in qty_values],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='RM IN QTY',
            x=materials,
            y=rm_values,
            marker_color='#ffc107',
            opacity=0.8,
            text=[f'{v:.1f}' for v in rm_values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="QTY vs RM IN QTY Comparison (Top 10 by Value)",
            xaxis_title="Material Code",
            yaxis_title="Quantity",
            height=400,
            barmode='group',
            xaxis_tickangle=-45,
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=1
            ),
            yaxis=dict(
                tickformat='.1f'
            )
        )
        
        return fig
    
    def create_scatter_plot(self, analysis_data):
        """Create scatter plot of QTY vs RM IN QTY"""
        qty_values = [item['QTY'] for item in analysis_data]
        rm_values = [item['RM IN QTY'] for item in analysis_data]
        colors = [self.status_colors[item['Status']] for item in analysis_data]
        materials = [item['Material'] for item in analysis_data]
        
        fig = go.Figure()
        
        # Group by status for legend
        for status in self.status_colors.keys():
            status_data = [item for item in analysis_data if item['Status'] == status]
            if status_data:
                fig.add_trace(go.Scatter(
                    x=[item['RM IN QTY'] for item in status_data],
                    y=[item['QTY'] for item in status_data],
                    mode='markers',
                    name=status,
                    marker=dict(color=self.status_colors[status], size=8),
                    text=[item['Material'] for item in status_data],
                    hovertemplate='<b>%{text}</b><br>RM IN QTY: %{x:.1f}<br>Current QTY: %{y:.1f}<extra></extra>'
                ))
        
        # Add diagonal line
        max_val = max(max(qty_values) if qty_values else 0, max(rm_values) if rm_values else 0)
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Perfect Match',
            line=dict(dash='dash', color='black'),
            opacity=0.5,
            showlegend=True
        ))
        
        fig.update_layout(
            title="QTY vs RM IN QTY Scatter Plot",
            xaxis_title="RM IN QTY",
            yaxis_title="Current QTY",
            height=400,
            xaxis=dict(
                tickformat='.1f'
            ),
            yaxis=dict(
                tickformat='.1f'
            )
        )
        
        return fig
    
    def create_variance_histogram(self, analysis_data):
        """Create histogram of variance distribution"""
        variances = [item['Variance_%'] for item in analysis_data]
        
        fig = go.Figure(data=[go.Histogram(
            x=variances,
            nbinsx=20,
            marker_color='#6c757d',
            opacity=0.7,
            name='Variance Distribution'
        )])
        
        # Get tolerance from session state (default to 30 if not available)
        tolerance = getattr(st.session_state, 'tolerance', 30)
        
        fig.add_vline(x=tolerance, line_dash="dash", line_color="red", 
                     annotation_text=f"+{tolerance}%", annotation_position="top")
        fig.add_vline(x=-tolerance, line_dash="dash", line_color="red",
                     annotation_text=f"-{tolerance}%", annotation_position="top")
        fig.add_vline(x=0, line_color="green", annotation_text="Perfect Match", 
                     annotation_position="top")
        
        fig.update_layout(
            title="Variance Distribution",
            xaxis_title="Variance %",
            yaxis_title="Frequency",
            height=400,
            xaxis=dict(
                tickformat='.1f',
                ticksuffix='%'
            ),
            yaxis=dict(
                tickformat='.0f'
            )
        )
        
        return fig
    
    def create_stock_impact_chart(self, summary_data):
        """Create chart showing stock value impact by status"""
        statuses = list(summary_data.keys())
        values = [summary_data[status]['value'] for status in statuses]
        colors = [self.status_colors[status] for status in statuses]
        
        fig = go.Figure(data=[go.Bar(
            x=statuses,
            y=values,
            marker_color=colors,
            text=[f'₹{v:,}' for v in values],
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Stock Value Impact by Status",
            xaxis_title="Status",
            yaxis_title="Stock Value (₹)",
            height=400,
            xaxis=dict(
                tickangle=-15
            ),
            yaxis=dict(
                tickformat=',.0f'
            )
        )
        
        return fig
    
    def create_3d_scatter_plot(self, analysis_data):
        """Create 3D scatter plot with QTY, RM, and Stock Value"""
        qty_values = [item['QTY'] for item in analysis_data]
        rm_values = [item['RM IN QTY'] for item in analysis_data]
        stock_values = [item['Stock_Value'] for item in analysis_data]
        materials = [item['Material'] for item in analysis_data]
        vendors = [item['Vendor'] for item in analysis_data]
        statuses = [item['Status'] for item in analysis_data]
        
        fig = go.Figure()
        
        # Group by status for legend
        for status in self.status_colors.keys():
            status_data = [item for item in analysis_data if item['Status'] == status]
            if status_data:
                fig.add_trace(go.Scatter3d(
                    x=[item['QTY'] for item in status_data],
                    y=[item['RM IN QTY'] for item in status_data],
                    z=[item['Stock_Value'] for item in status_data],
                    mode='markers',
                    name=status,
                    marker=dict(
                        color=self.status_colors[status],
                        size=6,
                        opacity=0.8
                    ),
                    text=[f"Material: {item['Material']}<br>Vendor: {item['Vendor']}<br>Variance: {item['Variance_%']:.1f}%" 
                          for item in status_data],
                    hovertemplate='<b>%{text}</b><br>QTY: %{x:.1f}<br>RM: %{y:.1f}<br>Value: ₹%{z:,}<extra></extra>'
                ))
        
        fig.update_layout(
            title="3D Analysis: QTY vs RM vs Stock Value",
            scene=dict(
                xaxis_title="Current QTY",
                yaxis_title="RM IN QTY",
                zaxis_title="Stock Value (₹)",
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            height=600,
            width=800
        )
        
        return fig
    
    def create_3d_surface_plot(self, analysis_data):
        """Create 3D surface plot showing variance patterns"""
        df = pd.DataFrame(analysis_data)
        
        # Create grid for surface plot
        x = np.linspace(df['QTY'].min(), df['QTY'].max(), 20)
        y = np.linspace(df['RM IN QTY'].min(), df['RM IN QTY'].max(), 20)
        X, Y = np.meshgrid(x, y)
        
        # Calculate variance surface
        Z = ((X - Y) / Y) * 100
        Z = np.where(Y == 0, 0, Z)  # Handle division by zero
        
        fig = go.Figure(data=[go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Variance %")
        )])
        
        # Add actual data points
        fig.add_trace(go.Scatter3d(
            x=df['QTY'],
            y=df['RM IN QTY'],
            z=df['Variance_%'],
            mode='markers',
            marker=dict(
                size=4,
                color='black',
                opacity=0.8
            ),
            name='Actual Data',
            hovertemplate='Material: %{text}<br>QTY: %{x:.1f}<br>RM: %{y:.1f}<br>Variance: %{z:.1f}%<extra></extra>',
            text=df['Material']
        ))
        
        fig.update_layout(
            title="3D Variance Surface Analysis",
            scene=dict(
                xaxis_title="Current QTY",
                yaxis_title="RM IN QTY",
                zaxis_title="Variance %",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600,
            width=800
        )
        
        return fig
    
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

# Initialize the analyzer
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = InventoryAnalyzer()

analyzer = st.session_state.analyzer

# App Title
st.markdown('<h1 class="main-header">🚀 Enhanced 3D Inventory Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for file upload and settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Inventory File",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with inventory data"
    )
    
    # Tolerance setting
    tolerance = st.slider(
        "Variance Tolerance (%)",
        min_value=5,
        max_value=50,
        value=30,
        step=5,
        help="Acceptable variance percentage for 'Within Norms' classification"
    )
    st.session_state.tolerance = tolerance
    
    # Use sample data option
    use_sample = st.checkbox("Use Sample Data", value=True if not uploaded_file else False)
    
    st.markdown("---")
    st.markdown("### 📝 File Format Requirements")
    st.markdown("""
    Your file should contain these columns:
    - **Material/Part Number**: Unique identifier
    - **QTY/Quantity**: Current stock quantity
    - **RM/RM IN QTY**: Required/target quantity
    - **Stock Value** (optional): Value in currency
    - **Vendor** (optional): Supplier information
    - **Description** (optional): Item description
    """)

# Load and process data
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        inventory_data = analyzer.standardize_inventory_data(df)
        if not inventory_data:
            st.error("Could not process the uploaded file. Please check the format.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()
elif use_sample:
    inventory_data = analyzer.load_sample_data()
else:
    st.info("Please upload a file or use sample data to begin analysis.")
    st.stop()

# Process the data
processed_data, summary_data = analyzer.process_data(inventory_data, tolerance)
vendor_summary = analyzer.get_vendor_summary(processed_data)

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "📈 2D Analysis", 
    "🌐 3D Visualization", 
    "🏢 Vendor Analysis", 
    "📋 Data Table"
])

with tab1:
    st.header("📊 Inventory Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_items = len(processed_data)
        st.metric("Total Items", total_items)
    
    with col2:
        total_value = sum(item['Stock_Value'] for item in processed_data)
        st.metric("Total Stock Value", f"₹{total_value:,}")
    
    with col3:
        excess_items = summary_data['Excess Inventory']['count']
        st.metric("Excess Items", excess_items, delta=f"{(excess_items/total_items*100):.1f}%")
    
    with col4:
        short_items = summary_data['Short Inventory']['count']
        st.metric("Short Items", short_items, delta=f"{(short_items/total_items*100):.1f}%")
    
    # Status cards
    st.subheader("📈 Status Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card status-normal">
            <h3>✅ Within Norms</h3>
            <h2>{summary_data['Within Norms']['count']} items</h2>
            <p>Value: ₹{summary_data['Within Norms']['value']:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card status-excess">
            <h3>📈 Excess Inventory</h3>
            <h2>{summary_data['Excess Inventory']['count']} items</h2>
            <p>Value: ₹{summary_data['Excess Inventory']['value']:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card status-short">
            <h3>📉 Short Inventory</h3>
            <h2>{summary_data['Short Inventory']['count']} items</h2>
            <p>Value: ₹{summary_data['Short Inventory']['value']:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main charts
    st.subheader("📊 Key Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_stock = analyzer.create_stock_value_chart(processed_data)
        st.plotly_chart(fig_stock, use_container_width=True)
    
    with col2:
        fig_impact = analyzer.create_stock_impact_chart(summary_data)
        st.plotly_chart(fig_impact, use_container_width=True)
    
    # Variance histogram
    fig_hist = analyzer.create_variance_histogram(processed_data)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.header("📈 2D Analysis")
    
    st.subheader("📈 Overview Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
        # Stock Impact Chart
        fig_stock_impact = analyzer.create_stock_impact_chart(summary_data)
        st.plotly_chart(fig_stock_impact, use_container_width=True)
    
    # Section 2: Value Analysis Charts
    st.subheader("💰 Value Analysis")
    col1, col2 = st.columns(2)
     with col1:
            # Top 10 Parts by Stock Value
            fig_stock_value = analyzer.create_stock_value_chart(processed_data)
            st.plotly_chart(fig_stock_value, use_container_width=True)
        
        with col2:
            # Top 10 Materials by Variance
            fig_variance = analyzer.create_variance_chart(processed_data)
            st.plotly_chart(fig_variance, use_container_width=True)
        
        # Section 3: Comparison Analysis
        st.subheader("⚖️ Comparison Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # QTY vs RM Comparison Chart
            fig_comparison = analyzer.create_comparison_chart(processed_data)
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            # QTY vs RM Scatter Plot
            fig_scatter_plot = analyzer.create_scatter_plot(processed_data)
            st.plotly_chart(fig_scatter_plot, use_container_width=True)
        
        # Section 4: Distribution Analysis
        st.subheader("📊 Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Variance Distribution Histogram
            fig_variance_hist = analyzer.create_variance_histogram(processed_data)
            st.plotly_chart(fig_variance_hist, use_container_width=True)
        
        with col2:
            # Enhanced QTY vs RM scatter plot with better formatting
            df_2d = pd.DataFrame(processed_data)
            fig_scatter = px.scatter(
                df_2d,
                x='RM IN QTY',
                y='QTY',
                color='Status',
                size='Stock_Value',
                hover_data=['Material', 'Description', 'Vendor', 'Variance_%'],
                title="Current QTY vs Required QTY (Enhanced)",
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
            
            # Update layout with better tick formatting
            fig_scatter.update_layout(
                xaxis=dict(
                    tickformat='.0f',
                    title='Required Quantity (RM IN QTY)'
                ),
                yaxis=dict(
                    tickformat='.0f',
                    title='Current Quantity (QTY)'
                ),
                height=400
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Section 5: Vendor Performance Analysis
        st.subheader("🏢 Vendor Performance Analysis")
        
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
        
        # Update layout with better formatting
        fig_vendor_bar.update_layout(
            xaxis=dict(
                tickangle=-45,
                title='Vendor'
            ),
            yaxis=dict(
                tickformat='.0f',
                title='Number of Items'
            ),
            height=400
        )
        
        st.plotly_chart(fig_vendor_bar, use_container_width=True)

with tab3:
    st.header("🌐 3D Visualization")
    
    st.subheader("🎯 3D Scatter Plot")
    fig_3d_scatter = analyzer.create_3d_scatter_plot(processed_data)
    st.plotly_chart(fig_3d_scatter, use_container_width=True)
    
    st.subheader("🌊 3D Surface Analysis")
    fig_3d_surface = analyzer.create_3d_surface_plot(processed_data)
    st.plotly_chart(fig_3d_surface, use_container_width=True)

with tab4:
    st.header("🏢 Vendor Analysis")
    
    # Vendor filter
    st.markdown('<div class="vendor-filter">', unsafe_allow_html=True)
    all_vendors = list(vendor_summary.keys())
    selected_vendors = st.multiselect(
        "Filter by Vendor:",
        options=all_vendors,
        default=all_vendors,
        help="Select vendors to include in the analysis"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if selected_vendors:
        # Filter data based on selected vendors
        filtered_data = [item for item in processed_data if item['Vendor'] in selected_vendors]
        
        # Vendor summary table
        st.subheader("📊 Vendor Summary")
        vendor_df = pd.DataFrame([
            {
                'Vendor': vendor,
                'Total Parts': data['total_parts'],
                'Total QTY': f"{data['total_qty']:.1f}",
                'Total RM': f"{data['total_rm']:.1f}",
                'Total Value': f"₹{data['total_value']:,}",
                'Short Parts': data['short_parts'],
                'Excess Parts': data['excess_parts'],
                'Normal Parts': data['normal_parts']
            }
            for vendor, data in vendor_summary.items()
            if vendor in selected_vendors
        ])
        st.dataframe(vendor_df, use_container_width=True)
        
        # Vendor-specific charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Vendor parts distribution
            vendor_parts = [vendor_summary[v]['total_parts'] for v in selected_vendors]
            fig_vendor_parts = go.Figure(data=[go.Bar(
                x=selected_vendors,
                y=vendor_parts,
                marker_color='#17a2b8',
                text=vendor_parts,
                textposition='outside'
            )])
            fig_vendor_parts.update_layout(
                title="Parts Distribution by Vendor",
                xaxis_title="Vendor",
                yaxis_title="Number of Parts",
                height=400
            )
            st.plotly_chart(fig_vendor_parts, use_container_width=True)
        
        with col2:
            # Vendor value distribution
            vendor_values = [vendor_summary[v]['total_value'] for v in selected_vendors]
            fig_vendor_value = go.Figure(data=[go.Bar(
                x=selected_vendors,
                y=vendor_values,
                marker_color='#28a745',
                text=[f'₹{v:,}' for v in vendor_values],
                textposition='outside'
            )])
            fig_vendor_value.update_layout(
                title="Stock Value Distribution by Vendor",
                xaxis_title="Vendor",
                yaxis_title="Stock Value (₹)",
                height=400
            )
            st.plotly_chart(fig_vendor_value, use_container_width=True)
        
        # Vendor status breakdown
        st.subheader("📈 Vendor Status Breakdown")
        
        # Create stacked bar chart for vendor status
        short_counts = [vendor_summary[v]['short_parts'] for v in selected_vendors]
        excess_counts = [vendor_summary[v]['excess_parts'] for v in selected_vendors]
        normal_counts = [vendor_summary[v]['normal_parts'] for v in selected_vendors]
        
        fig_vendor_status = go.Figure()
        
        fig_vendor_status.add_trace(go.Bar(
            name='Short Inventory',
            x=selected_vendors,
            y=short_counts,
            marker_color='#dc3545'
        ))
        
        fig_vendor_status.add_trace(go.Bar(
            name='Excess Inventory',
            x=selected_vendors,
            y=excess_counts,
            marker_color='#007bff'
        ))
        
        fig_vendor_status.add_trace(go.Bar(
            name='Within Norms',
            x=selected_vendors,
            y=normal_counts,
            marker_color='#28a745'
        ))
        
        fig_vendor_status.update_layout(
            title="Inventory Status by Vendor",
            xaxis_title="Vendor",
            yaxis_title="Number of Parts",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig_vendor_status, use_container_width=True)
        
        # Filtered 3D visualization
        if len(filtered_data) > 0:
            st.subheader("🌐 3D Analysis (Filtered by Vendor)")
            fig_3d_filtered = analyzer.create_3d_scatter_plot(filtered_data)
            st.plotly_chart(fig_3d_filtered, use_container_width=True)
    
    else:
        st.warning("Please select at least one vendor to view the analysis.")

with tab5:
    st.header("📋 Detailed Data Table")
    
    # Search and filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("🔍 Search Materials", "")
    
    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            options=['All'] + list(analyzer.status_colors.keys())
        )
    
    with col3:
        vendor_filter = st.selectbox(
            "Filter by Vendor",
            options=['All'] + list(vendor_summary.keys())
        )
    
    # Filter data based on search and filters
    filtered_table_data = processed_data.copy()
    
    if search_term:
        filtered_table_data = [
            item for item in filtered_table_data
            if search_term.lower() in item['Material'].lower() or 
               search_term.lower() in item['Description'].lower()
        ]
    
    if status_filter != 'All':
        filtered_table_data = [
            item for item in filtered_table_data
            if item['Status'] == status_filter
        ]
    
    if vendor_filter != 'All':
        filtered_table_data = [
            item for item in filtered_table_data
            if item['Vendor'] == vendor_filter
        ]
    
    # Create DataFrame for display
    if filtered_table_data:
        table_df = pd.DataFrame([
            {
                'Material': item['Material'],
                'Description': item['Description'],
                'Vendor': item['Vendor'],
                'Current QTY': f"{item['QTY']:.1f}",
                'RM IN QTY': f"{item['RM IN QTY']:.1f}",
                'Variance %': f"{item['Variance_%']:.1f}%",
                'Variance Value': f"{item['Variance_Value']:.1f}",
                'Status': item['Status'],
                'Stock Value': f"₹{item['Stock_Value']:,}"
            }
            for item in filtered_table_data
        ])
        
        # Style the dataframe
        def color_status(val):
            if val == 'Within Norms':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'Excess Inventory':
                return 'background-color: #cce5ff; color: #004085'
            elif val == 'Short Inventory':
                return 'background-color: #f8d7da; color: #721c24'
            return ''
        
        styled_df = table_df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Export options
        st.subheader("📥 Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = table_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"inventory_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create Excel file
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                table_df.to_excel(writer, sheet_name='Inventory Analysis', index=False)
            
            st.download_button(
                label="Download as Excel",
                data=buffer.getvalue(),
                file_name=f"inventory_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    else:
        st.info("No data matches the current filters.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>📊 Enhanced 3D Inventory Analysis Dashboard | Built with ❤️ using Streamlit & Plotly</p>
        <p>Data processed: {total_items} items | Last updated: {timestamp}</p>
    </div>
    """.format(
        total_items=len(processed_data),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ),
    unsafe_allow_html=True
)
