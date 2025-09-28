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

def parse_sheet_date(sheet_name):
    """Parse different date formats from sheet names"""
    # Try different formats
    formats = [
        '%d_%B_%Y',      # 15_August_2025
        '%d-%b-%Y',      # 12-Sep-2025
        '%d-%B-%Y',      # 12-September-2025
        '%d_%b_%Y',      # 15_Sep_2025
        '%Y-%m-%d',      # 2025-09-12
        '%m-%d-%Y',      # 09-12-2025
        '%d/%m/%Y',      # 12/09/2025
        '%m/%d/%Y',      # 09/12/2025
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(sheet_name, fmt)
        except ValueError:
            continue
    
    # If no format works, return None
    return None

def analyze_shareholder_changes(excel_file):
    try:
        xls = pd.ExcelFile(excel_file)
        sheet_names = xls.sheet_names
        
        # Parse and sort dates properly
        sheet_dates = []
        for sheet_name in sheet_names:
            parsed_date = parse_sheet_date(sheet_name)
            if parsed_date:
                sheet_dates.append((sheet_name, parsed_date))
            else:
                st.warning(f"Could not parse date from sheet name: '{sheet_name}'. Skipping this sheet.")
        
        if len(sheet_dates) < 3:
            raise ValueError("At least three sheets with valid dates are required for consecutive and overall comparisons.")
        
        # Sort by parsed date
        sheet_dates.sort(key=lambda x: x[1])
        dates = [sheet_name for sheet_name, _ in sheet_dates]
        
        st.info(f"Processing sheets in chronological order: {dates}")
        
        df_dict = {date: pd.read_excel(xls, sheet_name=date) for date in dates}
        
        id_col = 'PAN NO'
        name_col = 'NAME'
        holding_col = 'HOLDING '
        
        for date, df in df_dict.items():
            if id_col not in df.columns or name_col not in df.columns or holding_col not in df.columns:
                raise ValueError(f"Sheet '{date}' is missing one or more required columns: {id_col}, {name_col}, {holding_col}")
        
        all_data = []
        for date in dates:
            df = df_dict[date]
            df = df.groupby([id_col, name_col])[holding_col].sum().reset_index()
            df['Date'] = date
            all_data.append(df)
        
        all_df = pd.concat(all_data, ignore_index=True)
        
        names = all_df[[id_col, name_col]].drop_duplicates().set_index(id_col)[name_col]
        
        changes = []
        
        # ALL Consecutive comparisons (not just first two)
        for i in range(len(dates) - 1):
            date1 = dates[i]
            date2 = dates[i + 1]
            date_label = f"{date1} to {date2}"
            
            h1_df = df_dict[date1].groupby(id_col)[holding_col].sum()
            h2_df = df_dict[date2].groupby(id_col)[holding_col].sum()
            
            all_ids = set(h1_df.index) | set(h2_df.index)
            
            for id_ in all_ids:
                hold1 = h1_df.get(id_, 0)
                hold2 = h2_df.get(id_, 0)
                
                if hold1 == hold2:
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
                
                name = names.get(id_, 'Unknown')
                
                changes.append({
                    'Name': name,
                    'Action': action,
                    'Change Amount': change_amount,
                    'Date Transition': date_label
                })
        
        # Overall comparison (first to last)
        date1 = dates[0]
        date2 = dates[-1]
        date_label = f"Overall_{date1} to {date2}"
        
        h1_df = df_dict[date1].groupby(id_col)[holding_col].sum()
        h2_df = df_dict[date2].groupby(id_col)[holding_col].sum()
        
        all_ids = set(h1_df.index) | set(h2_df.index)
        
        for id_ in all_ids:
            hold1 = h1_df.get(id_, 0)
            hold2 = h2_df.get(id_, 0)
            
            if hold1 == hold2:
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
            
            name = names.get(id_, 'Unknown')
            
            changes.append({
                'Name': name,
                'Action': action,
                'Change Amount': change_amount,
                'Date Transition': date_label
            })
        
        changes_df = pd.DataFrame(changes)
        
        if changes_df.empty:
            return None, None, None, None, None, None, None, "No changes detected."
        
        # Pivot table for output
        pivot_df = changes_df.pivot_table(
            index=['Name', 'Action'],
            columns='Date Transition',
            values='Change Amount',
            aggfunc='first'
        ).reset_index()
        pivot_df = pivot_df.fillna(0)
        
        # Reorder columns: consecutive first (in chronological order), overall last
        date_columns = [col for col in pivot_df.columns if col not in ['Name', 'Action']]
        consecutive_labels = [f"{dates[i]} to {dates[i+1]}" for i in range(len(dates)-1)]
        overall_labels = [col for col in date_columns if col.startswith('Overall_')]
        
        # Order: Name, Action, consecutive transitions (chronological), overall
        ordered_columns = ['Name', 'Action'] + consecutive_labels + overall_labels
        pivot_df = pivot_df[ordered_columns]
        
        # Aggregate counts for visualization
        counts = changes_df.groupby(['Date Transition', 'Action']).size().unstack(fill_value=0)
        
        # Order date pairs: consecutive first, then overall
        date_pairs = consecutive_labels + overall_labels
        date_pairs = [dp for dp in date_pairs if dp in counts.index]
        
        increases = counts.get('increase', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        decreases = counts.get('decrease', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        exits = counts.get('exit', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        entries = counts.get('entry', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        
        # Create Plotly chart for web display
        fig = go.Figure()
        
        x = date_pairs
        width = 0.2
        
        fig.add_trace(go.Bar(
            x=x,
            y=increases,
            name='Increase',
            marker_color='#4CAF50',
            offset=-1.5*width,
            width=width,
            text=[int(i) if i > 0 else '' for i in increases],
            textposition='auto',
            hovertemplate='%{y} shareholders<br>Date: %{x}<extra>Increase</extra>'
        ))
        fig.add_trace(go.Bar(
            x=x,
            y=decreases,
            name='Decrease',
            marker_color='#F44336',
            offset=-0.5*width,
            width=width,
            text=[int(i) if i > 0 else '' for i in decreases],
            textposition='auto',
            hovertemplate='%{y} shareholders<br>Date: %{x}<extra>Decrease</extra>'
        ))
        fig.add_trace(go.Bar(
            x=x,
            y=exits,
            name='Exit',
            marker_color='#2196F3',
            offset=0.5*width,
            width=width,
            text=[int(i) if i > 0 else '' for i in exits],
            textposition='auto',
            hovertemplate='%{y} shareholders<br>Date: %{x}<extra>Exit</extra>'
        ))
        fig.add_trace(go.Bar(
            x=x,
            y=entries,
            name='Entry',
            marker_color='#FFC107',
            offset=1.5*width,
            width=width,
            text=[int(i) if i > 0 else '' for i in entries],
            textposition='auto',
            hovertemplate='%{y} shareholders<br>Date: %{x}<extra>Entry</extra>'
        ))
        
        fig.update_layout(
            title='Shareholder Changes Across Dates',
            xaxis_title='Date Transition',
            yaxis_title='Number of Shareholders',
            barmode='group',
            xaxis_tickangle=-45,
            legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
            margin=dict(b=150),
            showlegend=True,
            plot_bgcolor='white',
            font=dict(family="Arial", size=12)
        )
        
        return pivot_df, fig, None, increases, decreases, exits, entries, date_pairs
    
    except ValueError as e:
        return None, None, None, None, None, None, None, f"Error: {str(e)}"
    except Exception as e:
        return None, None, None, None, None, None, None, f"An unexpected error occurred: {str(e)}"

def generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, output):
    try:
        plt.figure(figsize=(12, 8))  # Made larger to accommodate more bars
        x = np.arange(len(date_pairs))
        width = 0.2
        
        bars1 = plt.bar(x - 1.5*width, increases, width, label='Increase', color='#4CAF50')
        bars2 = plt.bar(x - 0.5*width, decreases, width, label='Decrease', color='#F44336')
        bars3 = plt.bar(x + 0.5*width, exits, width, label='Exit', color='#2196F3')
        bars4 = plt.bar(x + 1.5*width, entries, width, label='Entry', color='#FFC107')
        
        plt.xlabel('Date Transition')
        plt.ylabel('Number of Shareholders')
        plt.title('Shareholder Changes Across Dates')
        plt.xticks(x, date_pairs, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Add labels to bars
        for bars, heights in [(bars1, increases), (bars2, decreases), (bars3, exits), (bars4, entries)]:
            for bar, height in zip(bars, heights):
                if height > 0:  # Only label bars with non-zero height
                    plt.text(bar.get_x() + bar.get_width() / 2, height,
                             f'{int(height)}', ha='center', va='bottom')
        
        plt.savefig(output, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        return True, None
    except Exception as e:
        return False, f"Error generating matplotlib plot: {str(e)}"

def generate_pdf(pivot_df, fig, increases, decreases, exits, entries, date_pairs):
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save pivot table to temporary Excel file
            temp_excel = os.path.join(tmpdirname, "temp_changes.xlsx")
            with pd.ExcelWriter(temp_excel, engine='openpyxl') as writer:
                pivot_df.to_excel(writer, index=False, sheet_name='Changes')
            
            # Generate matplotlib plot
            plot_path = os.path.join(tmpdirname, "plot.png")
            success, error = generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, plot_path)
            if not success:
                return None, error
            
            # Initialize PDF
            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Set font
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Shareholder Changes Report", ln=True, align='C')
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, "Generated by Streamlit App", ln=True, align='C')
            pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
            pdf.ln(10)
            
            # Add pivot table
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Summary Table", ln=True, align='L')
            pdf.set_font("Arial", "", 10)
            
            # Calculate column widths with more balanced distribution
            page_width = pdf.w - 2 * pdf.l_margin
            num_data_cols = len(pivot_df.columns) - 2  # Exclude Name and Action
            col_widths = [page_width * 0.25, page_width * 0.15] + [page_width * 0.60 / num_data_cols] * num_data_cols
            
            # Header with text wrapping on the same row, matching Overall column height
            pdf.set_fill_color(200, 200, 200)  # Light gray for header
            start_y = pdf.get_y()
            # Calculate height for Overall column
            overall_col_idx = len(pivot_df.columns) - 1
            overall_text = str(pivot_df.columns[overall_col_idx])
            overall_width = col_widths[overall_col_idx]
            avg_char_width = pdf.get_string_width('a')
            overall_text_width = pdf.get_string_width(overall_text)
            overall_num_lines = max(1, int(overall_text_width / overall_width) + 1)
            max_height = 5 * overall_num_lines  # 5mm per line, based on Overall column
            
            current_x = pdf.get_x()
            for i, (col, width) in enumerate(zip(pivot_df.columns, col_widths)):
                text = str(col)
                # Fill background for the entire column height
                pdf.rect(current_x, start_y, width, max_height, 'F')
                # Render text centered in the fixed height
                pdf.set_xy(current_x, start_y)  # Reset to start of column
                pdf.multi_cell(width, 5, text, border=1, align='C', ln=0)
                current_x += width  # Move to next column position
            
            pdf.set_y(start_y + max_height)  # Move to the row below
            
            # Table rows with color coding based on Action
            for _, row in pivot_df.iterrows():
                action = row['Action']
                if action in ['increase', 'entry']:
                    pdf.set_fill_color(76, 175, 80)  # Green for increase/entry
                elif action in ['decrease', 'exit']:
                    pdf.set_fill_color(244, 67, 54)  # Red for decrease/exit
                else:
                    pdf.set_fill_color(240, 240, 240)  # Light gray for others
                
                for val, width in zip(row, col_widths):
                    val_str = str(val)
                    if len(val_str) > 30:  # Truncate long text
                        val_str = val_str[:27] + "..."
                    pdf.cell(width, 10, val_str, border=1, align='L', fill=True)
                pdf.ln()
            
            # Start new page for Visualization
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Visualization", ln=True, align='L')
            
            # Add plot directly below the header
            pdf.image(plot_path, x=10, y=pdf.get_y(), w=page_width)
            
            # Save PDF
            pdf_path = os.path.join(tmpdirname, "report.pdf")
            pdf.output(pdf_path)
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
            return pdf_data, None
    
    except Exception as e:
        return None, f"Error generating PDF: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Shareholder Changes Analyzer", layout="wide")

st.title("Shareholder Changes Analyzer")
st.markdown("""
Upload an Excel file with at least three sheets representing different dates. 
The sheets will be automatically sorted by date, and the app will analyze all consecutive transitions plus an overall comparison.

**Supported date formats in sheet names:**
- `15_August_2025` or `15_Sep_2025`  
- `12-Sep-2025` or `12-September-2025`
- `2025-09-12` or `09-12-2025`
- `12/09/2025` or `09/12/2025`

Click the 'Analyze' button to process the data and generate a summary table, downloadable Excel, interactive visualization, and a PDF report.
""")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if 'files_generated' not in st.session_state:
    st.session_state.files_generated = False
    st.session_state.excel_data = None
    st.session_state.png_data = None
    st.session_state.pdf_data = None
    st.session_state.pivot_df = None
    st.session_state.fig = None

if uploaded_file is not None:
    if st.button("Analyze"):
        with st.spinner("Analyzing data..."):
            pivot_df, fig, error, increases, decreases, exits, entries, date_pairs = analyze_shareholder_changes(uploaded_file)
        
        if error:
            st.error(error)
        else:
            st.success("Analysis complete!")
            
            st.session_state.pivot_df = pivot_df
            st.session_state.fig = fig
            
            # Generate and store Excel data
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                pivot_df.to_excel(writer, index=False, sheet_name='Changes')
            output.seek(0)
            st.session_state.excel_data = output.getvalue()
            
            # Generate and store PNG data
            plot_output = io.BytesIO()
            success, error = generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, plot_output)
            if success:
                plot_output.seek(0)
                st.session_state.png_data = plot_output.getvalue()
            else:
                st.error(error)
            
            # Generate and store PDF data
            with st.spinner("Generating PDF report..."):
                pdf_data, pdf_error = generate_pdf(pivot_df, fig, increases, decreases, exits, entries, date_pairs)
            if pdf_error:
                st.error(pdf_error)
            else:
                st.session_state.pdf_data = pdf_data
            
            st.session_state.files_generated = True
    
    if st.session_state.files_generated:
        st.header("Summary Table")
        st.dataframe(st.session_state.pivot_df, use_container_width=True)
        
        # Download Excel
        st.download_button(
            label="Download Changes Excel",
            data=st.session_state.excel_data,
            file_name="shareholder_changes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )
        
        # Display Plot (Plotly)
        st.header("Shareholder Changes Visualization")
        st.markdown("Hover over the bars to see the number of shareholders and date transitions.")
        st.plotly_chart(st.session_state.fig, use_container_width=True)
        
        # Download Plot (Matplotlib)
        if st.session_state.png_data:
            st.download_button(
                label="Download Plot PNG",
                data=st.session_state.png_data,
                file_name="shareholder_changes_plot.png",
                mime="image/png",
                key="download_plot"
            )
        
        # Download PDF
        if st.session_state.pdf_data:
            st.download_button(
                label="Download PDF Report",
                data=st.session_state.pdf_data,
                file_name="shareholder_changes_report.pdf",
                mime="application/pdf",
                key="download_pdf"
            )
else:
    st.info("Please upload an Excel file and click 'Analyze' to begin.")
