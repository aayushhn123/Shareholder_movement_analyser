import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import tempfile
from fpdf import FPDF
from datetime import datetime
import re

def parse_date_from_sheet_name(sheet_name):
    """Extract date from sheet name with flexible parsing"""
    try:
        # Common date patterns
        date_patterns = [
            r'(\d{1,2})[_\-\s]*([A-Za-z]+)[_\-\s]*(\d{4})',  # 15_August_2024
            r'(\d{4})[_\-\s]*(\d{1,2})[_\-\s]*(\d{1,2})',    # 2024-08-15
            r'(\d{1,2})[_\-\s]*(\d{1,2})[_\-\s]*(\d{4})',    # 15-08-2024
            r'([A-Za-z]+)[_\-\s]*(\d{1,2})[_\-\s]*(\d{4})',  # August_15_2024
        ]
        
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'october': 10, 'oct': 10,
            'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, sheet_name.lower())
            if match:
                groups = match.groups()
                
                # Pattern 1: day_month_year
                if len(groups) == 3 and groups[1].isalpha():
                    day, month_str, year = groups
                    month = month_map.get(month_str.lower())
                    if month:
                        return datetime(int(year), month, int(day))
                
                # Pattern 2: year_month_day
                elif len(groups) == 3 and len(groups[0]) == 4:
                    year, month, day = groups
                    return datetime(int(year), int(month), int(day))
                
                # Pattern 3: day_month_year (numeric)
                elif len(groups) == 3 and len(groups[2]) == 4:
                    day, month, year = groups
                    return datetime(int(year), int(month), int(day))
                
                # Pattern 4: month_day_year
                elif len(groups) == 3 and groups[0].isalpha():
                    month_str, day, year = groups
                    month = month_map.get(month_str.lower())
                    if month:
                        return datetime(int(year), month, int(day))
        
        # If no pattern matches, try to use the sheet name as is
        return sheet_name
        
    except Exception:
        return sheet_name

def analyze_shareholder_changes(excel_file):
    try:
        xls = pd.ExcelFile(excel_file)
        sheet_names = xls.sheet_names
        
        if len(sheet_names) < 2:
            raise ValueError("At least two sheets are required for comparison.")
        
        # Parse and sort dates
        sheet_date_map = {}
        for sheet in sheet_names:
            parsed_date = parse_date_from_sheet_name(sheet)
            sheet_date_map[sheet] = parsed_date
        
        # Sort sheets by parsed date (datetime objects first, then strings)
        sorted_sheets = sorted(sheet_names, key=lambda x: (
            isinstance(sheet_date_map[x], str),  # Strings come after datetime objects
            sheet_date_map[x]
        ))
        
        dates = sorted_sheets
        df_dict = {date: pd.read_excel(xls, sheet_name=date) for date in dates}
        
        # Auto-detect column names with flexible matching
        id_col, name_col, holding_col = detect_columns(df_dict[dates[0]])
        
        # Validate columns exist in all sheets
        for date, df in df_dict.items():
            missing_cols = []
            if id_col not in df.columns:
                missing_cols.append(id_col)
            if name_col not in df.columns:
                missing_cols.append(name_col)
            if holding_col not in df.columns:
                missing_cols.append(holding_col)
            
            if missing_cols:
                raise ValueError(f"Sheet '{date}' is missing columns: {', '.join(missing_cols)}")
        
        # Clean and prepare data
        for date in dates:
            df = df_dict[date]
            # Convert holding to numeric, handling any text/formatting issues
            df[holding_col] = pd.to_numeric(df[holding_col], errors='coerce').fillna(0)
            df_dict[date] = df
        
        all_data = []
        for date in dates:
            df = df_dict[date]
            df = df.groupby([id_col, name_col])[holding_col].sum().reset_index()
            df['Date'] = date
            all_data.append(df)
        
        all_df = pd.concat(all_data, ignore_index=True)
        names = all_df[[id_col, name_col]].drop_duplicates().set_index(id_col)[name_col]
        
        changes = []
        comparison_pairs = []
        
        # Generate all consecutive date pairs
        for i in range(len(dates) - 1):
            comparison_pairs.append((dates[i], dates[i + 1], f"{dates[i]} to {dates[i + 1]}"))
        
        # Add overall comparison (first to last)
        if len(dates) > 2:
            comparison_pairs.append((dates[0], dates[-1], f"Overall_{dates[0]} to {dates[-1]}"))
        
        # Process all comparisons
        for date1, date2, date_label in comparison_pairs:
            h1_df = df_dict[date1].groupby(id_col)[holding_col].sum()
            h2_df = df_dict[date2].groupby(id_col)[holding_col].sum()
            
            all_ids = set(h1_df.index) | set(h2_df.index)
            
            for id_ in all_ids:
                hold1 = h1_df.get(id_, 0)
                hold2 = h2_df.get(id_, 0)
                
                if abs(hold1 - hold2) < 0.01:  # Handle floating point precision
                    continue
                
                if hold1 == 0 and hold2 > 0:
                    action = 'entry'
                    change_amount = hold2
                elif hold2 > hold1:
                    action = 'increase'
                    change_amount = hold2 - hold1
                elif hold2 < hold1:
                    action = 'exit' if hold2 == 0 else 'decrease'
                    change_amount = hold2 - hold1
                else:
                    continue
                
                name = names.get(id_, 'Unknown')
                
                changes.append({
                    'Name': name,
                    'Action': action,
                    'Change Amount': change_amount,
                    'Date Transition': date_label,
                    'From Date': date1,
                    'To Date': date2
                })
        
        changes_df = pd.DataFrame(changes)
        
        if changes_df.empty:
            return None, None, None, None, None, None, None, "No changes detected.", comparison_pairs
        
        # Create pivot table
        pivot_df = changes_df.pivot_table(
            index=['Name', 'Action'],
            columns='Date Transition',
            values='Change Amount',
            aggfunc='first'
        ).reset_index()
        pivot_df = pivot_df.fillna(0)
        
        # Reorder columns to show consecutive comparisons first, then overall
        date_columns = [col for col in pivot_df.columns if col not in ['Name', 'Action']]
        consecutive_cols = [col for col in date_columns if not col.startswith('Overall_')]
        overall_cols = [col for col in date_columns if col.startswith('Overall_')]
        
        ordered_columns = consecutive_cols + overall_cols
        pivot_df = pivot_df[['Name', 'Action'] + ordered_columns]
        
        # Generate visualization data
        counts = changes_df.groupby(['Date Transition', 'Action']).size().unstack(fill_value=0)
        
        # Order date pairs for visualization
        date_pairs = consecutive_cols + overall_cols
        date_pairs = [dp for dp in date_pairs if dp in counts.index]
        
        increases = counts.get('increase', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        decreases = counts.get('decrease', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        exits = counts.get('exit', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        entries = counts.get('entry', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        
        # Create Plotly chart
        fig = create_plotly_chart(date_pairs, increases, decreases, exits, entries)
        
        return pivot_df, fig, None, increases, decreases, exits, entries, date_pairs, None
    
    except ValueError as e:
        return None, None, None, None, None, None, None, f"Error: {str(e)}", None
    except Exception as e:
        return None, None, None, None, None, None, None, f"An unexpected error occurred: {str(e)}", None

def detect_columns(df):
    """Auto-detect ID, Name, and Holding columns with flexible matching"""
    columns = [col.upper().strip() for col in df.columns]
    
    # Patterns for different column types
    id_patterns = ['PAN', 'ID', 'PANNO', 'PAN_NO', 'IDENTIFIER', 'CODE']
    name_patterns = ['NAME', 'SHAREHOLDER', 'HOLDER', 'INVESTOR', 'PERSON']
    holding_patterns = ['HOLDING', 'SHARES', 'SHARE', 'AMOUNT', 'QUANTITY', 'QTY', 'UNITS']
    
    id_col = name_col = holding_col = None
    
    # Find ID column
    for col in df.columns:
        col_upper = col.upper().strip()
        if any(pattern in col_upper for pattern in id_patterns):
            id_col = col
            break
    
    # Find Name column
    for col in df.columns:
        col_upper = col.upper().strip()
        if any(pattern in col_upper for pattern in name_patterns):
            name_col = col
            break
    
    # Find Holding column
    for col in df.columns:
        col_upper = col.upper().strip()
        if any(pattern in col_upper for pattern in holding_patterns):
            holding_col = col
            break
    
    # If auto-detection fails, use default patterns or first available
    if not id_col:
        id_col = 'PAN NO' if 'PAN NO' in df.columns else df.columns[0]
    if not name_col:
        name_col = 'NAME' if 'NAME' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0]
    if not holding_col:
        holding_col = 'HOLDING ' if 'HOLDING ' in df.columns else df.columns[-1]
    
    return id_col, name_col, holding_col

def create_plotly_chart(date_pairs, increases, decreases, exits, entries):
    """Create enhanced Plotly chart with better styling"""
    fig = go.Figure()
    
    colors = {
        'increase': '#4CAF50',  # Green
        'decrease': '#F44336',  # Red
        'exit': '#2196F3',      # Blue
        'entry': '#FFC107'      # Amber
    }
    
    width = max(0.15, 0.8 / len(date_pairs)) if date_pairs else 0.2
    
    traces = [
        ('Increase', increases, colors['increase'], -1.5),
        ('Decrease', decreases, colors['decrease'], -0.5),
        ('Exit', exits, colors['exit'], 0.5),
        ('Entry', entries, colors['entry'], 1.5)
    ]
    
    for name, values, color, offset in traces:
        fig.add_trace(go.Bar(
            x=date_pairs,
            y=values,
            name=name,
            marker_color=color,
            offset=offset*width,
            width=width,
            text=[int(v) if v > 0 else '' for v in values],
            textposition='auto',
            textfont=dict(size=10, color='white' if name in ['Decrease'] else 'black'),
            hovertemplate=f'%{{y}} shareholders<br>Period: %{{x}}<extra>{name}</extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Shareholder Changes Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Time Period',
        yaxis_title='Number of Shareholders',
        barmode='group',
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(b=120, t=80),
        showlegend=True,
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white',
        font=dict(family="Arial", size=12),
        height=500
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, output):
    """Generate matplotlib plot with improved styling"""
    try:
        fig_size = (max(10, len(date_pairs) * 1.5), 8)
        plt.figure(figsize=fig_size)
        
        x = np.arange(len(date_pairs))
        width = 0.18
        
        colors = ['#4CAF50', '#F44336', '#2196F3', '#FFC107']
        
        bars = []
        bars.append(plt.bar(x - 1.5*width, increases, width, label='Increase', color=colors[0]))
        bars.append(plt.bar(x - 0.5*width, decreases, width, label='Decrease', color=colors[1]))
        bars.append(plt.bar(x + 0.5*width, exits, width, label='Exit', color=colors[2]))
        bars.append(plt.bar(x + 1.5*width, entries, width, label='Entry', color=colors[3]))
        
        plt.xlabel('Time Period', fontsize=12)
        plt.ylabel('Number of Shareholders', fontsize=12)
        plt.title('Shareholder Changes Analysis', fontsize=16, fontweight='bold')
        plt.xticks(x, date_pairs, rotation=45, ha='right')
        plt.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # Add value labels on bars
        for bar_group, values in zip(bars, [increases, decreases, exits, entries]):
            for bar, value in zip(bar_group, values):
                if value > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{int(value)}', ha='center', va='bottom', fontsize=9)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        return True, None
    except Exception as e:
        plt.close()
        return False, f"Error generating matplotlib plot: {str(e)}"

def generate_excel_data(pivot_df):
    """Generate Excel data with enhanced formatting"""
    try:
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl', mode='w') as writer:
            if pivot_df is not None and not pivot_df.empty:
                pivot_df.to_excel(writer, index=False, sheet_name='Changes')
                
                # Get workbook and worksheet
                workbook = writer.book
                worksheet = workbook['Changes']
                worksheet.sheet_state = 'visible'
                workbook.active = worksheet
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
            else:
                dummy_df = pd.DataFrame({'Message': ['No changes detected']})
                dummy_df.to_excel(writer, index=False, sheet_name='NoChanges')
                
                workbook = writer.book
                worksheet = workbook['NoChanges']
                worksheet.sheet_state = 'visible'
                workbook.active = worksheet
        
        output.seek(0)
        return output.getvalue(), None
        
    except Exception as e:
        return None, f"Error generating Excel file: {str(e)}"

def generate_pdf(pivot_df, fig, increases, decreases, exits, entries, date_pairs):
    """Generate PDF with improved layout and formatting"""
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Generate matplotlib plot
            plot_path = os.path.join(tmpdirname, "plot.png")
            success, error = generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, plot_path)
            if not success:
                return None, error
            
            # Initialize PDF
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Header
            pdf.set_font("Arial", "B", 18)
            pdf.cell(0, 15, "Shareholder Changes Analysis Report", ln=True, align='C')
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%B %d, %Y at %H:%M')}", ln=True, align='C')
            pdf.ln(10)
            
            # Summary statistics
            if pivot_df is not None and not pivot_df.empty:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 8, "Executive Summary", ln=True)
                pdf.set_font("Arial", "", 10)
                
                total_changes = len(pivot_df)
                total_increases = len(pivot_df[pivot_df['Action'] == 'increase'])
                total_decreases = len(pivot_df[pivot_df['Action'] == 'decrease'])
                total_entries = len(pivot_df[pivot_df['Action'] == 'entry'])
                total_exits = len(pivot_df[pivot_df['Action'] == 'exit'])
                
                pdf.cell(0, 6, f"• Total Changes Detected: {total_changes}", ln=True)
                pdf.cell(0, 6, f"• Shareholding Increases: {total_increases}", ln=True)
                pdf.cell(0, 6, f"• Shareholding Decreases: {total_decreases}", ln=True)
                pdf.cell(0, 6, f"• New Entries: {total_entries}", ln=True)
                pdf.cell(0, 6, f"• Complete Exits: {total_exits}", ln=True)
                pdf.ln(10)
                
                # Detailed table
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, "Detailed Changes Table", ln=True)
                pdf.set_font("Arial", "", 9)
                
                # Calculate dynamic column widths
                page_width = pdf.w - 2 * pdf.l_margin
                num_cols = len(pivot_df.columns)
                
                # Allocate more space for Name column, less for others
                name_width = page_width * 0.3
                action_width = page_width * 0.15
                remaining_width = page_width - name_width - action_width
                data_col_width = remaining_width / (num_cols - 2) if num_cols > 2 else remaining_width
                
                col_widths = [name_width, action_width] + [data_col_width] * (num_cols - 2)
                
                # Table header
                pdf.set_fill_color(220, 220, 220)
                for col, width in zip(pivot_df.columns, col_widths):
                    col_text = str(col)
                    if len(col_text) > 20:
                        col_text = col_text[:17] + "..."
                    pdf.cell(width, 8, col_text, border=1, align='C', fill=True)
                pdf.ln()
                
                # Table rows (limit to first 30 rows for space)
                display_df = pivot_df.head(30)
                for _, row in display_df.iterrows():
                    action = row['Action']
                    if action in ['increase', 'entry']:
                        pdf.set_fill_color(200, 255, 200)  # Light green
                    elif action in ['decrease', 'exit']:
                        pdf.set_fill_color(255, 200, 200)  # Light red
                    else:
                        pdf.set_fill_color(255, 255, 255)  # White
                    
                    for val, width in zip(row, col_widths):
                        val_str = str(val)
                        if len(val_str) > 25:
                            val_str = val_str[:22] + "..."
                        pdf.cell(width, 6, val_str, border=1, align='C', fill=True)
                    pdf.ln()
                
                if len(pivot_df) > 30:
                    pdf.set_font("Arial", "I", 9)
                    pdf.cell(0, 6, f"Note: Showing first 30 rows of {len(pivot_df)} total changes. Download Excel for complete data.", ln=True)
            
            # New page for visualization
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Visualization", ln=True)
            
            # Add plot
            plot_width = min(page_width, 250)
            pdf.image(plot_path, x=15, y=pdf.get_y() + 5, w=plot_width)
            
            # Save PDF
            pdf_path = os.path.join(tmpdirname, "report.pdf")
            pdf.output(pdf_path)
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
            return pdf_data, None
    
    except Exception as e:
        return None, f"Error generating PDF: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Advanced Shareholder Changes Analyzer", layout="wide")

st.title("🏢 Advanced Shareholder Changes Analyzer")
st.markdown("""
**Enhanced Multi-Date Analysis Tool**

Upload an Excel file with multiple sheets representing different dates. This tool will:
- ✅ **Auto-detect** column names (ID, Name, Holdings)
- 📅 **Parse dates** from sheet names automatically  
- 📊 **Generate comparisons** between consecutive dates + overall changes
- 📈 **Create visualizations** and downloadable reports

**Supported date formats in sheet names:** 
`15_August_2024`, `2024-08-15`, `Aug_15_2024`, etc.
""")

# File upload with enhanced info
uploaded_file = st.file_uploader(
    "Upload your Excel file", 
    type=["xlsx", "xls"],
    help="File should contain multiple sheets with dates in sheet names"
)

# Initialize session state
if 'files_generated' not in st.session_state:
    st.session_state.files_generated = False
    st.session_state.excel_data = None
    st.session_state.excel_error = None
    st.session_state.png_data = None
    st.session_state.pdf_data = None
    st.session_state.pivot_df = None
    st.session_state.fig = None
    st.session_state.analysis_info = None

if uploaded_file is not None:
    # Show file info
    with st.expander("📋 File Information", expanded=False):
        try:
            xls = pd.ExcelFile(uploaded_file)
            st.write(f"**Sheets found:** {len(xls.sheet_names)}")
            
            for i, sheet in enumerate(xls.sheet_names, 1):
                parsed_date = parse_date_from_sheet_name(sheet)
                if isinstance(parsed_date, datetime):
                    st.write(f"{i}. `{sheet}` → {parsed_date.strftime('%B %d, %Y')}")
                else:
                    st.write(f"{i}. `{sheet}` → Could not parse date")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_button = st.button("🔍 Analyze Shareholder Changes", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.files_generated = False
            st.rerun()
    
    if analyze_button:
        with st.spinner("🔍 Analyzing shareholder data..."):
            result = analyze_shareholder_changes(uploaded_file)
            pivot_df, fig, error, increases, decreases, exits, entries, date_pairs, comparison_pairs = result
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success("✅ Analysis completed successfully!")
            
            # Store results
            st.session_state.pivot_df = pivot_df
            st.session_state.fig = fig
            st.session_state.analysis_info = {
                'periods': len(date_pairs) if date_pairs else 0,
                'total_changes': len(pivot_df) if pivot_df is not None else 0
            }
            
            # Generate files
            with st.spinner("📊 Generating downloadable files..."):
                # Excel
                excel_data, excel_error = generate_excel_data(pivot_df)
                st.session_state.excel_data = excel_data
                st.session_state.excel_error = excel_error
                
                # PNG
                plot_output = io.BytesIO()
                success, error = generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, plot_output)
                if success:
                    plot_output.seek(0)
                    st.session_state.png_data = plot_output.getvalue()
                else:
                    st.error(f"Plot generation error: {error}")
                
                # PDF
                pdf_data, pdf_error = generate_pdf(pivot_df, fig, increases, decreases, exits, entries, date_pairs)
                if pdf_error:
                    st.error(f"PDF generation error: {pdf_error}")
                else:
                    st.session_state.pdf_data = pdf_data
            
            st.session_state.files_generated = True
    
    # Display results
    if st.session_state.files_generated:
        # Summary metrics
        if st.session_state.analysis_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Analysis Periods", st.session_state.analysis_info['periods'])
            with col2:
                st.metric("Total Changes", st.session_state.analysis_info['total_changes'])
            with col3:
                downloads_available = sum([
                    bool(st.session_state.excel_data),
                    bool(st.session_state.png_data),
                    bool(st.session_state.pdf_data)
                ])
                st.metric("Downloads Ready", f"{downloads_available}/3")
        
        # Summary table
        st.header("📋 Summary Table")
        if st.session_state.pivot_df is not None and not st.session_state.pivot_df.empty:
            st.dataframe(st.session_state.pivot_df, use_container_width=True, height=400)
            
            # Quick stats
            with st.expander("📊 Quick Statistics"):
                df = st.session_state.pivot_df
                actions = df['Action'].value_counts()
                for action, count in actions.items():
                    st.write(f"**{action.title()}:** {count} shareholders")
        else:
            st.info("ℹ️ No changes detected in the shareholder data.")
        
        # Download buttons
        st.header("⬇️ Downloads")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.excel_data and not st.session_state.excel_error:
                st.download_button(
                    label="📊 Download Excel Report",
                    data=st.session_state.excel_data,
                    file_name=f"shareholder_changes_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.error("❌ Excel generation failed")
        
        with col2:
            if st.session_state.png_data:
                st.download_button(
                    label="📈 Download Chart (PNG)",
                    data=st.session_state.png_data,
                    file_name=f"shareholder_changes_chart_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.error("❌ Chart generation failed")
        
        with col3:
            if st.session_state.pdf_data:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=st.session_state.pdf_data,
                    file_name=f"shareholder_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("❌ PDF generation failed")
        
        # Interactive visualization
        st.header("📈 Interactive Visualization")
        if st.session_state.fig:
            st.markdown("*Hover over bars to see detailed information*")
            st.plotly_chart(st.session_state.fig, use_container_width=True)
        else:
            st.error("Visualization not available")
        
        # Additional insights
        if st.session_state.pivot_df is not None and not st.session_state.pivot_df.empty:
            with st.expander("🔍 Advanced Insights", expanded=False):
                df = st.session_state.pivot_df
                
                # Top changes
                st.subheader("Top Changes by Action")
                for action in ['increase', 'decrease', 'entry', 'exit']:
                    action_df = df[df['Action'] == action]
                    if not action_df.empty:
                        st.write(f"**{action.title()}** ({len(action_df)} shareholders)")
                        # Show top 5 for each action
                        display_df = action_df.head(5)[['Name', 'Action']]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

else:
    st.info("📁 Please upload an Excel file to begin analysis")
    
    # Help section
    with st.expander("❓ Help & Requirements", expanded=True):
        st.markdown("""
        ### File Requirements
        - **Format:** Excel file (.xlsx or .xls)
        - **Minimum sheets:** 2 (for comparison)
        - **Sheet naming:** Include dates (e.g., "15_August_2024", "2024-08-15")
        
        ### Required Columns (auto-detected)
        - **ID column:** PAN, PAN_NO, ID, etc.
        - **Name column:** NAME, SHAREHOLDER, HOLDER, etc.  
        - **Holdings column:** HOLDING, SHARES, AMOUNT, etc.
        
        ### What the tool does
        1. **Parses dates** from sheet names automatically
        2. **Compares** consecutive time periods + overall changes
        3. **Categorizes** changes as: Increase, Decrease, Entry, Exit
        4. **Generates** summary tables, charts, and reports
        5. **Exports** results in Excel, PNG, and PDF formats
        
        ### Supported Date Formats in Sheet Names
        - `15_August_2024` or `15-August-2024`
        - `2024-08-15` or `2024_08_15`  
        - `Aug_15_2024` or `August_15_2024`
        - And many other common formats...
        """)
        
        st.markdown("---")
        st.markdown("*💡 **Pro tip:** The more sheets you have, the more detailed your analysis will be!*")
