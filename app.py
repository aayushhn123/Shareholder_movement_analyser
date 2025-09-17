import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import io

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
        for i in range(2):  # First two transitions
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
        ordered_columns = sorted(date_columns, key=lambda x: ('15 to 29' not in x, x))  # Ensure 15-29 is last
        pivot_df = pivot_df[['Name', 'Action'] + ordered_columns]
        
        # Aggregate counts for visualization
        counts = changes_df.groupby(['Date Transition', 'Action']).size().unstack(fill_value=0)
        date_pairs = sorted(counts.index, key=lambda x: ('15 to 29' not in x, x))  # Ensure 15-29 is last
        increases = counts.get('increase', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        decreases = counts.get('decrease', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        exits = counts.get('exit', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        entries = counts.get('entry', pd.Series(0, index=counts.index)).reindex(date_pairs, fill_value=0).tolist()
        
        # Create grouped bar chart
        x = np.arange(len(date_pairs))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - 1.5*width, increases, width, label='Increase', color='#4CAF50')
        bars2 = ax.bar(x - 0.5*width, decreases, width, label='Decrease', color='#F44336')
        bars3 = ax.bar(x + 0.5*width, exits, width, label='Exit', color='#2196F3')
        bars4 = ax.bar(x + 1.5*width, entries, width, label='Entry', color='#FFC107')
        
        ax.set_xlabel('Date Transition')
        ax.set_ylabel('Number of Shareholders')
        ax.set_title('Shareholder Changes Across Dates')
        ax.set_xticks(x)
        ax.set_xticklabels(date_pairs, rotation=45, ha='right')
        ax.legend()
        
        # Add data labels
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}',
                            ha='center', va='bottom', fontsize=10)
        
        # Add tooltips (note: mplcursors may not work in Streamlit's static display; consider Plotly for interactivity)
        cursor = mplcursors.cursor([bars1, bars2, bars3, bars4], hover=True)
        cursor.connect(
            "add",
            lambda sel: sel.annotation.set_text(
                f"{sel.artist.get_label()}: {int(sel.target[1])} shareholders\n"
                f"Date: {date_pairs[int(sel.target[0] // 1)]}"
            )
        )
        
        plt.tight_layout()
        
        return pivot_df, fig, None
    
    except ValueError as e:
        return None, None, f"Error: {str(e)}"
    except Exception as e:
        return None, None, f"An unexpected error occurred: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Shareholder Changes Analyzer", layout="wide")

st.title("Shareholder Changes Analyzer")
st.markdown("""
Upload an Excel file with at least three sheets representing different dates. 
The app will analyze changes in shareholder holdings and provide a summary table, downloadable Excel, and a visualization.
""")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
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
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
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
        st.markdown("Note: Hover tooltips may not be interactive in this display. For full interactivity, download the plot or use a local environment.")
        st.pyplot(fig)
        
        # Download Plot
        plot_output = io.BytesIO()
        fig.savefig(plot_output, format="png")
        plot_output.seek(0)
        st.download_button(
            label="Download Plot PNG",
            data=plot_output,
            file_name="shareholder_changes_plot.png",
            mime="image/png",
            key="download_plot"
        )
else:
    st.info("Please upload an Excel file to begin.")
