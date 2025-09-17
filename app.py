import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import io
import base64
import subprocess
import os
import tempfile

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
            return None, None, "No changes detected."
        
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
        
        # Create grouped bar chart with Plotly
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
        
        return pivot_df, fig, None
    
    except ValueError as e:
        return None, None, f"Error: {str(e)}"
    except Exception as e:
        return None, None, f"An unexpected error occurred: {str(e)}"

def generate_pdf(pivot_df, fig):
    # Creating temporary directory for LaTeX processing
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Save plot as PNG
        plot_path = os.path.join(tmpdirname, "plot.png")
        fig.write_image(plot_path, format="png")
        
        # Convert table to LaTeX format
        table_latex = "\\begin{tabular}{ll" + "r" * (len(pivot_df.columns) - 2) + "}\n\\hline\n"
        table_latex += " & ".join([col.replace("_", "\\_") for col in pivot_df.columns]) + " \\\\\n\\hline\n"
        for _, row in pivot_df.iterrows():
            table_latex += " & ".join([str(val).replace("_", "\\_") for val in row]) + " \\\\\n"
        table_latex += "\\hline\n\\end{tabular}"
        
        # LaTeX document
        latex_content = f"""
        \\documentclass{{article}}
        \\usepackage{{geometry}}
        \\usepackage{{graphicx}}
        \\usepackage{{booktabs}}
        \\usepackage{{parskip}}
        \\usepackage[utf8]{{inputenc}}
        \\usepackage{{amsmath}}
        \\usepackage{{fontenc}}
        \\usepackage{{lmodern}}
        \\geometry{{a4paper, margin=1in}}
        \\title{{Shareholder Changes Report}}
        \\author{{Generated by Streamlit App}}
        \\date{{September 2025}}
        \\begin{{document}}
        \\maketitle
        \\section*{{Summary Table}}
        \\begin{{center}}
        {table_latex}
        \\end{{center}}
        \\section*{{Visualization}}
        \\begin{{center}}
        \\includegraphics[width=\\textwidth]{{{os.path.basename(plot_path)}}}
        \\end{{center}}
        \\end{{document}}
        """
        
        # Write LaTeX file
        tex_path = os.path.join(tmpdirname, "report.tex")
        with open(tex_path, "w") as f:
            f.write(latex_content)
        
        # Compile LaTeX to PDF
        try:
            subprocess.run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", tex_path],
                cwd=tmpdirname,
                check=True,
                capture_output=True
            )
            pdf_path = os.path.join(tmpdirname, "report.pdf")
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
            return pdf_data
        except subprocess.CalledProcessError as e:
            return None, f"Error generating PDF: {e.stderr.decode()}"
        finally:
            # Clean up LaTeX auxiliary files
            subprocess.run(["latexmk", "-c"], cwd=tmpdirname, capture_output=True)

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
            pivot_df, fig, error = analyze_shareholder_changes(uploaded_file)
        
        if error:
            st.error(error)
        else:
            st.success("Analysis complete!")
            
            st.header("Summary Table")
            st.dataframe(pivot_df, use_container_width=True)
            
            # Download Excel
            output = io.BytesIO()
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
            
            # Display Plot
            st.header("Shareholder Changes Visualization")
            st.markdown("Hover over the bars to see the number of shareholders and date transitions.")
            st.plotly_chart(fig, use_container_width=True)
            
            # Download Plot
            plot_output = io.BytesIO()
            fig.write_image(plot_output, format="png")
            plot_output.seek(0)
            st.download_button(
                label="Download Plot PNG",
                data=plot_output,
                file_name="shareholder_changes_plot.png",
                mime="image/png",
                key="download_plot"
            )
            
            # Generate and Download PDF
            pdf_data = generate_pdf(pivot_df, fig)
            if isinstance(pdf_data, tuple):
                st.error(pdf_data[1])
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
