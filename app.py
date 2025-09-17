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
        dates = sorted(sheet_names)
        if len(dates) < 3:
            raise ValueError("At least three sheets are required for 15-22, 22-29, and 15-29 comparisons.")
        
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
        # Consecutive comparisons (15-22, 22-29)
        for i in range(2):
            date1 = dates[i]
            date2 = dates[i + 1]
            date_label = f"{date1[-2:]} to {date2[-2:]}" if date1.startswith('20') else f"{date1} to {date2}"
            
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
        
        # Overall comparison (15-29)
        date1 = dates[0]
        date2 = dates[-1]
        date_label = f"{date1[-2:]} to {date2[-2:]}" if date1.startswith('20') else f"{date1} to {date2}"
        
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
        
        # Reorder columns to ensure 15-22, 22-29, 15-29
        date_columns = [col for col in pivot_df.columns if col not in ['Name', 'Action']]
        ordered_columns = sorted(date_columns, key=lambda x: ('15 to 29' not in x, x))
        pivot_df = pivot_df[['Name', 'Action'] + ordered_columns]
        
        # Aggregate counts for visualization
        counts = changes_df.groupby(['Date Transition', 'Action']).size().unstack(fill_value=0)
        date_pairs = sorted(counts.index, key=lambda x: ('15 to 29' not in x, x))
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
            
            # Calculate column widths
            page_width = pdf.w - 2 * pdf.l_margin
            col_widths = [page_width * 0.3, page_width * 0.15] + [page_width * 0.55 / len(pivot_df.columns[2:])] * len(pivot_df.columns[2:])
            
            # Header
            for col, width in zip(pivot_df.columns, col_widths):
                pdf.set_fill_color(200, 200, 200)  # Light gray for header
                pdf.cell(width, 10, str(col), border=1, align='C', fill=True)
            pdf.ln()
            
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
            
            # Add plot
            pdf.ln(10)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Visualization", ln=True, align='L')
            pdf.image(plot_path, x=10, y=None, w=page_width)
            
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

if uploaded_file is not None:
    if st.button("Analyze"):
        with st.spinner("Analyzing data..."):
            pivot_df, fig, error, increases, decreases, exits, entries, date_pairs = analyze_shareholder_changes(uploaded_file)
        
        if error:
            st.error(error)
        else:
            st.success("Analysis complete!")
            
            st.header("Summary Table")
            st.dataframe(pivot_df, use_container_width=True)
            
            # Download Excel
            output = io.BytesIO()
            try:
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    pivot_df.to_excel(writer, index=False, sheet_name='Changes')
                output.seek(0)
                st.download_button(
                    label="Download Changes Excel",
                    data=output,
                    file_name="shareholder_changes.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
            except Exception as e:
                st.error(f"Error generating Excel: {str(e)}")
            
            # Display Plot (Plotly)
            st.header("Shareholder Changes Visualization")
            st.markdown("Hover over the bars to see the number of shareholders and date transitions.")
            st.plotly_chart(fig, use_container_width=True)
            
            # Download Plot (Matplotlib)
            plot_output = io.BytesIO()
            try:
                success, error = generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, plot_output)
                if success:
                    plot_output.seek(0)
                    st.download_button(
                        label="Download Plot PNG",
                        data=plot_output,
                        file_name="shareholder_changes_plot.png",
                        mime="image/png",
                        key="download_plot"
                    )
                else:
                    st.error(error)
            except Exception as e:
                st.error(f"Error saving plot image: {str(e)}")
            
            # Generate and Download PDF
            with st.spinner("Generating PDF report..."):
                pdf_data, pdf_error = generate_pdf(pivot_df, fig, increases, decreases, exits, entries, date_pairs)
            if pdf_error:
                st.error(pdf_error)
            else:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_data,
                    file_name="shareholder_changes_report.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )
else:
    st.info("Please upload an Excel file and click 'Analyze' to begin.")
