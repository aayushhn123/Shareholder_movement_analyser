import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import tempfile
from fpdf import FPDF

def analyze_shareholder_changes(excel_file):
    try:
        xls = pd.ExcelFile(excel_file)
        sheet_names = xls.sheet_names
        dates = sorted(sheet_names)  # assumes sheet names are dates or sortable
        if len(dates) < 2:
            raise ValueError("At least two sheets are required for comparison.")

        # Read all sheets into dictionary
        df_dict = {date: pd.read_excel(xls, sheet_name=date) for date in dates}

        id_col = 'PAN NO'
        name_col = 'NAME'
        holding_col = 'HOLDING '

        # Validate required columns
        for date, df in df_dict.items():
            if id_col not in df.columns or name_col not in df.columns or holding_col not in df.columns:
                raise ValueError(f"Sheet '{date}' is missing one or more required columns: {id_col}, {name_col}, {holding_col}")

        # Clean and standardize
        all_data = []
        for date in dates:
            df = df_dict[date]
            df = df.groupby([id_col, name_col])[holding_col].sum().reset_index()
            df['Date'] = date
            all_data.append(df)
        all_df = pd.concat(all_data, ignore_index=True)
        names = all_df[[id_col, name_col]].drop_duplicates().set_index(id_col)[name_col]

        changes = []

        # Consecutive comparisons (dynamic)
        for i in range(len(dates) - 1):
            date1, date2 = dates[i], dates[i + 1]
            date_label = f"{date1} to {date2}"

            h1_df = df_dict[date1].groupby(id_col)[holding_col].sum()
            h2_df = df_dict[date2].groupby(id_col)[holding_col].sum()
            all_ids = set(h1_df.index) | set(h2_df.index)

            for id_ in all_ids:
                hold1, hold2 = h1_df.get(id_, 0), h2_df.get(id_, 0)
                if hold1 == hold2:
                    continue

                if hold1 == 0 and hold2 > 0:
                    action, change_amount = 'entry', hold2
                elif hold2 > hold1:
                    action, change_amount = 'increase', hold2 - hold1
                elif hold2 < hold1:
                    action, change_amount = ('exit' if hold2 == 0 else 'decrease'), hold2 - hold1

                name = names.get(id_, 'Unknown')
                changes.append({
                    'Name': name,
                    'Action': action,
                    'Change Amount': change_amount,
                    'Date Transition': date_label
                })

        # Overall comparison (first → last)
        if len(dates) > 2:
            date1, date2 = dates[0], dates[-1]
            date_label = f"Overall_{date1} to {date2}"
            h1_df = df_dict[date1].groupby(id_col)[holding_col].sum()
            h2_df = df_dict[date2].groupby(id_col)[holding_col].sum()
            all_ids = set(h1_df.index) | set(h2_df.index)

            for id_ in all_ids:
                hold1, hold2 = h1_df.get(id_, 0), h2_df.get(id_, 0)
                if hold1 == hold2:
                    continue

                if hold1 == 0 and hold2 > 0:
                    action, change_amount = 'entry', hold2
                elif hold2 > hold1:
                    action, change_amount = 'increase', hold2 - hold1
                elif hold2 < hold1:
                    action, change_amount = ('exit' if hold2 == 0 else 'decrease'), hold2 - hold1

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

        # Pivot table
        pivot_df = changes_df.pivot_table(
            index=['Name', 'Action'],
            columns='Date Transition',
            values='Change Amount',
            aggfunc='first'
        ).reset_index().fillna(0)

        # Aggregate counts for visualization
        counts = changes_df.groupby(['Date Transition', 'Action']).size().unstack(fill_value=0)
        date_pairs = list(counts.index)
        increases = counts.get('increase', pd.Series(0, index=counts.index)).tolist()
        decreases = counts.get('decrease', pd.Series(0, index=counts.index)).tolist()
        exits = counts.get('exit', pd.Series(0, index=counts.index)).tolist()
        entries = counts.get('entry', pd.Series(0, index=counts.index)).tolist()

        # Plotly chart
        fig = go.Figure()
        width = 0.2
        x = date_pairs
        fig.add_trace(go.Bar(x=x, y=increases, name='Increase', marker_color='#4CAF50',
                             offset=-1.5*width, width=width, text=[int(i) if i > 0 else '' for i in increases],
                             textposition='auto'))
        fig.add_trace(go.Bar(x=x, y=decreases, name='Decrease', marker_color='#F44336',
                             offset=-0.5*width, width=width, text=[int(i) if i > 0 else '' for i in decreases],
                             textposition='auto'))
        fig.add_trace(go.Bar(x=x, y=exits, name='Exit', marker_color='#2196F3',
                             offset=0.5*width, width=width, text=[int(i) if i > 0 else '' for i in exits],
                             textposition='auto'))
        fig.add_trace(go.Bar(x=x, y=entries, name='Entry', marker_color='#FFC107',
                             offset=1.5*width, width=width, text=[int(i) if i > 0 else '' for i in entries],
                             textposition='auto'))

        fig.update_layout(
            title='Shareholder Changes Across Dates',
            xaxis_title='Date Transition',
            yaxis_title='Number of Shareholders',
            barmode='group',
            xaxis_tickangle=-45,
            showlegend=True,
            plot_bgcolor='white'
        )

        return pivot_df, fig, None, increases, decreases, exits, entries, date_pairs

    except Exception as e:
        return None, None, None, None, None, None, None, f"Error: {str(e)}"


def generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, output):
    try:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(date_pairs))
        width = 0.2
        
        bars1 = plt.bar(x - 1.5*width, increases, width, label='Increase', color='#4CAF50')
        bars2 = plt.bar(x - 0.5*width, decreases, width, label='Decrease', color='#F44336')
        bars3 = plt.bar(x + 0.5*width, exits, width, label='Exit', color='#2196F3')
        bars4 = plt.bar(x + 1.5*width, entries, width, label='Entry', color='#FFC107')
        
        plt.xlabel('Date Transition')
        plt.ylabel('Number of Shareholders')
        plt.title('Shareholder Changes Across Dates')
        plt.xticks(x, date_pairs, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Add labels to bars
        for bars, heights in [(bars1, increases), (bars2, decreases), (bars3, exits), (bars4, entries)]:
            for bar, height in zip(bars, heights):
                if height > 0:  # Only label bars with non-zero height
                    plt.text(bar.get_x() + bar.get_width() / 2, height,
                             f'{int(height)}', ha='center', va='bottom')
        
        plt.savefig(output, format='png', dpi=300)
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
            pdf.cell(0, 10, "September 2025", ln=True, align='C')
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
                pdf.multi_cell(width, max_height / overall_num_lines, text, border=1, align='C', ln=0)
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
