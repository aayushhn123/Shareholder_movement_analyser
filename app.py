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

# -----------------------------
# Helpers: parse sheet name -> date
# -----------------------------
def parse_sheet_date(sheet_name):
    """Parse different date formats from sheet names"""
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
    return None


# -----------------------------
# Core analysis
# -----------------------------
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
            df = df_dict[date].copy()
            df = df.groupby([id_col, name_col], as_index=False)[holding_col].sum()
            df['Date'] = date
            all_data.append(df)

        all_df = pd.concat(all_data, ignore_index=True)

        # Create a mapping of PAN NO to NAME
        names_dict = {}
        for _, row in all_df[[id_col, name_col]].drop_duplicates().iterrows():
            names_dict[row[id_col]] = row[name_col]

        changes = []

        # ALL Consecutive comparisons
        for i in range(len(dates) - 1):
            date1 = dates[i]
            date2 = dates[i + 1]
            date_label = f"{date1} to {date2}"

            # Dictionaries from grouped data
            h1_df = df_dict[date1].groupby(id_col, as_index=False)[holding_col].sum()
            h1_dict = dict(zip(h1_df[id_col], h1_df[holding_col]))

            h2_df = df_dict[date2].groupby(id_col, as_index=False)[holding_col].sum()
            h2_dict = dict(zip(h2_df[id_col], h2_df[holding_col]))

            all_ids = set(h1_dict.keys()) | set(h2_dict.keys())

            for id_ in all_ids:
                hold1 = h1_dict.get(id_, 0)
                hold2 = h2_dict.get(id_, 0)

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

                name = names_dict.get(id_, 'Unknown')

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

        h1_df = df_dict[date1].groupby(id_col, as_index=False)[holding_col].sum()
        h1_dict = dict(zip(h1_df[id_col], h1_df[holding_col]))

        h2_df = df_dict[date2].groupby(id_col, as_index=False)[holding_col].sum()
        h2_dict = dict(zip(h2_df[id_col], h2_df[holding_col]))

        all_ids = set(h1_dict.keys()) | set(h2_dict.keys())

        for id_ in all_ids:
            hold1 = h1_dict.get(id_, 0)
            hold2 = h2_dict.get(id_, 0)

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

            name = names_dict.get(id_, 'Unknown')

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

        # Reorder columns
        date_columns = [col for col in pivot_df.columns if col not in ['Name', 'Action']]
        consecutive_labels = [f"{dates[i]} to {dates[i+1]}" for i in range(len(dates)-1)]
        overall_labels = [col for col in date_columns if col.startswith('Overall_')]

        ordered_columns = ['Name', 'Action'] + consecutive_labels + overall_labels
        pivot_df = pivot_df[ordered_columns]

        # Aggregate counts for visualization
        counts = changes_df.groupby(['Date Transition', 'Action']).size().unstack(fill_value=0)

        date_pairs = consecutive_labels + overall_labels
        date_pairs = [dp for dp in date_pairs if dp in counts.index]

        increases = counts.get('increase', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        decreases = counts.get('decrease', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        exits = counts.get('exit', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        entries = counts.get('entry', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()

        # Plotly bar chart
        fig = go.Figure()
        x = date_pairs
        width = 0.2

        fig.add_trace(go.Bar(
            x=x, y=increases, name='Increase', marker_color='#4CAF50',
            offset=-1.5*width, width=width,
            text=[int(i) if i > 0 else '' for i in increases],
            textposition='auto',
            hovertemplate='%{y} shareholders<br>Date: %{x}<extra>Increase</extra>'
        ))
        fig.add_trace(go.Bar(
            x=x, y=decreases, name='Decrease', marker_color='#F44336',
            offset=-0.5*width, width=width,
            text=[int(i) if i > 0 else '' for i in decreases],
            textposition='auto',
            hovertemplate='%{y} shareholders<br>Date: %{x}<extra>Decrease</extra>'
        ))
        fig.add_trace(go.Bar(
            x=x, y=exits, name='Exit', marker_color='#2196F3',
            offset=0.5*width, width=width,
            text=[int(i) if i > 0 else '' for i in exits],
            textposition='auto',
            hovertemplate='%{y} shareholders<br>Date: %{x}<extra>Exit</extra>'
        ))
        fig.add_trace(go.Bar(
            x=x, y=entries, name='Entry', marker_color='#FFC107',
            offset=1.5*width, width=width,
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


# -----------------------------
# Matplotlib (for PDF image)
# -----------------------------
def generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, output):
    try:
        plt.figure(figsize=(12, 8))
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

        for bars, heights in [(bars1, increases), (bars2, decreases), (bars3, exits), (bars4, entries)]:
            for bar, height in zip(bars, heights):
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width() / 2, height,
                             f'{int(height)}', ha='center', va='bottom')

        plt.savefig(output, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        return True, None
    except Exception as e:
        return False, f"Error generating matplotlib plot: {str(e)}"


# -----------------------------
# PDF (styled like screenshot)
# -----------------------------
def generate_pdf(pivot_df, fig, increases, decreases, exits, entries, date_pairs):
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            # plot for last page
            plot_path = os.path.join(tmpdirname, "plot.png")
            success, error = generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, plot_path)
            if not success:
                return None, error

            pdf = FPDF(orientation='L', unit='mm', format='A4')
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Shareholder Changes Report", ln=True, align='C')
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%d-%B-%Y')}", ln=True, align='C')
            pdf.ln(6)

            # ----- Summary Table (navy header, wrapped, aligned, colored numbers)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Summary Table", ln=True, align='L')
            pdf.set_font("Arial", "", 10)

            # wrapped column names identical to Excel rendering
            disp_cols = []
            for c in pivot_df.columns:
                if c in ("Name", "Action"):
                    disp_cols.append(c)
                elif str(c).startswith("Overall_"):
                    base = str(c).replace("Overall_", "")
                    disp_cols.append("Overall\n" + base.replace(" to ", " to\n"))
                else:
                    disp_cols.append(str(c).replace(" to ", " to\n"))

            page_width = pdf.w - 2 * pdf.l_margin
            num_data_cols = len(pivot_df.columns) - 2
            col_widths = [page_width * 0.28, page_width * 0.12] + [page_width * 0.60 / max(1, num_data_cols)] * num_data_cols

            # Header styling (navy fill, white text)
            pdf.set_fill_color(11, 45, 107)  # 0B2D6B
            pdf.set_text_color(255, 255, 255)
            start_y = pdf.get_y()
            current_x = pdf.get_x()
            header_row_height = 8  # good height for two lines

            for text, width in zip(disp_cols, col_widths):
                pdf.set_xy(current_x, start_y)
                pdf.multi_cell(width, 4, text, border=1, align='C', fill=True)
                current_x += width

            pdf.set_y(start_y + header_row_height)
            pdf.set_text_color(0, 0, 0)

            # Body rows
            for _, row in pivot_df.iterrows():
                row_height = 8

                # Name (left)
                pdf.set_font("Arial", "", 10)
                pdf.cell(col_widths[0], row_height, str(row["Name"]), border=1, align='L')

                # Action (left, colored)
                action = str(row["Action"]).strip().lower()
                if action == 'entry':
                    pdf.set_text_color(255, 193, 7)
                elif action == 'exit':
                    pdf.set_text_color(33, 150, 243)
                elif action == 'increase':
                    pdf.set_text_color(76, 175, 80)
                elif action == 'decrease':
                    pdf.set_text_color(244, 67, 54)
                else:
                    pdf.set_text_color(0, 0, 0)
                pdf.cell(col_widths[1], row_height, row["Action"], border=1, align='L')
                pdf.set_text_color(0, 0, 0)

                # Numbers (centered; color by action; 0 black)
                for i, col in enumerate(pivot_df.columns[2:], start=2):
                    val = row[col]
                    try:
                        num = float(val)
                    except (TypeError, ValueError):
                        num = 0.0

                    if abs(num) > 0:
                        if action in ('increase', 'entry'):
                            pdf.set_text_color(76, 175, 80)
                        elif action in ('decrease', 'exit'):
                            pdf.set_text_color(244, 67, 54)
                        else:
                            pdf.set_text_color(0, 0, 0)
                    else:
                        pdf.set_text_color(0, 0, 0)

                    # integer-like formatting
                    txt = f"{int(num):,}" if float(num).is_integer() else f"{num:,.0f}"
                    pdf.cell(col_widths[i], row_height, txt, border=1, align='C')

                pdf.ln()
                pdf.set_text_color(0, 0, 0)

            # Visualization page
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Visualization", ln=True, align='L')
            page_width = pdf.w - 2 * pdf.l_margin
            pdf.image(plot_path, x=10, y=pdf.get_y(), w=page_width)

            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            return pdf_bytes, None

    except Exception as e:
        return None, f"Error generating PDF: {str(e)}"


# -----------------------------
# Streamlit App
# -----------------------------
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

            # -----------------------------
            # Excel export (styled like screenshot)
            # -----------------------------
            output = io.BytesIO()
            try:
                from openpyxl.styles import Font, Alignment, PatternFill
                from openpyxl.utils import get_column_letter

                # Build wrapped header labels (2 lines) — fixed (no f-string with \n in expression)
                display_df = pivot_df.copy()
                wrapped_cols = []
                for c in display_df.columns:
                    if c in ("Name", "Action"):
                        wrapped_cols.append(c)
                    elif str(c).startswith("Overall_"):
                        base = str(c).replace("Overall_", "")
                        wrapped_cols.append("Overall\n" + base.replace(" to ", " to\n"))
                    else:
                        wrapped_cols.append(str(c).replace(" to ", " to\n"))
                display_df.columns = wrapped_cols

                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    sheet = "Changes"
                    display_df.to_excel(writer, index=False, sheet_name=sheet, startrow=0)
                    ws = writer.sheets[sheet]

                    # Header style
                    header_fill = PatternFill("solid", fgColor="0B2D6B")  # navy
                    header_font = Font(bold=True, color="FFFFFF", size=12)
                    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
                    ws.row_dimensions[1].height = 30

                    for col_idx, _ in enumerate(display_df.columns, start=1):
                        cell = ws.cell(row=1, column=col_idx)
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = header_align

                    # Column widths
                    widths = []
                    for c in display_df.columns:
                        if c == "Name":
                            widths.append(38)
                        elif c == "Action":
                            widths.append(14)
                        else:
                            widths.append(18)

                    for i, w in enumerate(widths, start=1):
                        ws.column_dimensions[get_column_letter(i)].width = w

                    # Alignments
                    left_align = Alignment(horizontal="left", vertical="center")
                    center_align = Alignment(horizontal="center", vertical="center")

                    name_col_idx = 1
                    action_col_idx = 2
                    first_data_col_idx = 3
                    last_row = ws.max_row
                    last_col = ws.max_column

                    for r in range(2, last_row + 1):
                        # Name
                        ws.cell(row=r, column=name_col_idx).alignment = left_align

                        # Action (colored)
                        action_cell = ws.cell(row=r, column=action_col_idx)
                        action_cell.alignment = left_align
                        action_value = str(action_cell.value or "").strip().lower()
                        if action_value == "entry":
                            action_cell.font = Font(color="FFC107")
                        elif action_value == "exit":
                            action_cell.font = Font(color="2196F3")
                        elif action_value == "increase":
                            action_cell.font = Font(color="4CAF50")
                        elif action_value == "decrease":
                            action_cell.font = Font(color="F44336")

                        # Data cells
                        for c in range(first_data_col_idx, last_col + 1):
                            cell = ws.cell(row=r, column=c)
                            cell.alignment = center_align
                            cell.number_format = '#,##0;[Black]-#,##0'
                            try:
                                v = float(cell.value)
                            except (TypeError, ValueError):
                                v = 0.0
                            if abs(v) > 0:
                                if action_value in ("increase", "entry"):
                                    cell.font = Font(color="4CAF50")
                                elif action_value in ("decrease", "exit"):
                                    cell.font = Font(color="F44336")
                            else:
                                cell.font = Font(color="000000")

                    # Freeze header
                    ws.freeze_panes = "A2"

                output.seek(0)
                st.session_state.excel_data = output.getvalue()

            except Exception as e:
                st.error(f"Error generating Excel file: {str(e)}")
                st.session_state.excel_data = None

            # PNG for download (optional)
            plot_output = io.BytesIO()
            ok, err = generate_matplotlib_plot(increases, decreases, exits, entries, date_pairs, plot_output)
            if ok:
                plot_output.seek(0)
                st.session_state.png_data = plot_output.getvalue()
            else:
                st.error(err)

            # PDF
            with st.spinner("Generating PDF report..."):
                pdf_data, pdf_error = generate_pdf(pivot_df, st.session_state.fig, increases, decreases, exits, entries, date_pairs)
            if pdf_error:
                st.error(pdf_error)
            else:
                st.session_state.pdf_data = pdf_data

            st.session_state.files_generated = True

# -----------------------------
# UI after generation
# -----------------------------
if st.session_state.files_generated:
    st.header("Summary Table")
    st.dataframe(st.session_state.pivot_df, use_container_width=True)

    st.download_button(
        label="Download Changes Excel",
        data=st.session_state.excel_data,
        file_name="shareholder_changes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel"
    )

    st.header("Shareholder Changes Visualization")
    st.markdown("Hover over the bars to see the number of shareholders and date transitions.")
    st.plotly_chart(st.session_state.fig, use_container_width=True)

    if st.session_state.png_data:
        st.download_button(
            label="Download Plot PNG",
            data=st.session_state.png_data,
            file_name="shareholder_changes_plot.png",
            mime="image/png",
            key="download_plot"
        )

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
