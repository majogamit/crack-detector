import pandas as pd
from IPython.display import display, HTML
import os
import pdfkit
from pypdf import PdfMerger
import imgkit

def classify_wall_damage(crack_width):
    if crack_width <= 0.1:
        return "Negligible"
    elif 0.1 <= crack_width <= 1:
        return "Very slight"
    elif 1.1 <= crack_width  <= 5:
        return "Slight"
    elif 5 <= crack_width <= 15:
        return "Moderate"
    elif 15 <= crack_width <= 25:
        return "Severe"
    elif crack_width > 25:
        return "Very severe"
    else:
        return "Invalid input"
    

from collections import Counter

def generate_html_summary(crack_list):
    # Define the possible damage levels
    damage_levels = ["Negligible", "Very Slight", "Slight", "Moderate", "Severe", "Very Severe"]

    # Count the occurrences of each damage level
    string_counts = Counter(crack_list)

    # Build the HTML string
    html_summary = "<html>\n<body>\n"
    html_summary += "<h2>Summary of this batch</h2>\n"
    html_summary += "<p><strong>Number of Cracks Detected:</strong></p>\n"
    html_summary += "<ul>\n"

    # Append the damage level and count to the HTML string
    for level in damage_levels:
        count = string_counts.get(level, 0)
        html_summary += f"<li>{level} = {count}</li>\n"

    html_summary += "</ul>\n"
    html_summary += "</body>\n</html>"
    print(html_summary)
    return html_summary


def count_instance(result, filenames, uuid, width_list, orientation_list, image_path, reference, remark, damage):
        """
        Counts the instances in the result and generates a CSV with the counts.

        Parameters:
            result (list): List containing results for each instance.
            filenames (list): Corresponding filenames for each result.
            uuid (str): Unique ID for the output folder name.
            width_list (list): List containing width values for each instance.
            orientation_list (list): List containing orientation values for each instance.

        Returns:
            tuple: Path to the generated CSV and dataframe with counts.
        """
        # Initializing the dataframe
        print(damage)
        data = {
            'Index': [],
            'FileName': [],
            'Orientation': [],
            'Width (mm)': [],
            'Instance': [],
            'Damage Level': []
        }
        
        df_ref = pd.DataFrame({'Reference': [f'<img src="{ref}" width="640" >' for ref in reference]})


        df = pd.DataFrame(data)

        # Populate the dataframe with counts, width, and orientation
        for i, res in enumerate(result):
            instance_count = len(res)
            df.loc[i] = [i, os.path.basename(filenames[i]), orientation_list[i], width_list[i], instance_count, damage[i]]

        # Reorder columns
        df = df[['Index', 'FileName', 'Orientation', 'Width (mm)','Damage Level', 'Instance']]

        # Create a new dataframe (df2) with all columns from df
        df2 = df.copy()
        summary = generate_html_summary(damage)
        # Add another column for the image (modify as per your requirement)
        print("IMG PATHS")
        print(image_path)
        base_path = [os.path.basename(path) for path in image_path]
        df2['Image'] = base_path
        df2['Remarks'] = remark
        # convert your links to html tags 
        def path_to_image_html(path):
            return '<img src="'+ path + '" width="320" >'
        print("This executed 1")
        pd.set_option('display.max_colwidth', None)

        image_cols = ['Image']

        format_dict = {}
        for image_col in image_cols:
            format_dict[image_col] = path_to_image_html

        print("This executed 2")
        col_widths = [100, 50, 50, 50, 50, 120, 150] 
        df2 = df2.drop(df.columns[0], axis=1)

        # Create the HTML file
        df_html = df2.to_html(f'output/{uuid}/df_batch.html', escape=False, formatters=format_dict, col_space=col_widths, justify='left')
        df_refs = df_ref.to_html(f'output/{uuid}/df_ref.html', escape=False, justify='left')
        print("This executed 3")
        # Save the modified dataframe to a CSV file
        from bs4 import BeautifulSoup

        # Load the HTML file
        with open(f'output/{uuid}/df_ref.html', 'r') as file:
            html_content = file.read()

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find the table in the HTML (assuming there is only one table)
        table = soup.find('table')

        # Append the new summary HTML after the table
        table.insert_after(BeautifulSoup(summary, 'html.parser'))

        # Save the modified HTML to a new file
        with open(f'output/{uuid}/df_ref_summary.html', 'w') as file:
            file.write(str(soup))

        html_table = HTML(df2.to_html(escape=False))
        display(html_table)
              
        print('This executed 4')
        # new_parser = HtmlToDocx()
        # new_parser.parse_html_file(f'output/{uuid}/df_batch.html', f'output/{uuid}/report_batch')
        # new_parser.parse_html_file(f'output/{uuid}/df_ref_summary.html', f'output/{uuid}/report_ref')

        # convert(f"output/{uuid}/report_batch.docx", f"output/{uuid}/Mine.pdf")
        # pdfkit.from_file(f'output/{uuid}/df_batch.html', f'output/{uuid}/report_batch.pdf')
        # pdfkit.from_file(f'output/{uuid}/df_ref_summary.html', f'output/{uuid}/report_ref.pdf')

        print("This executed 5")

        # pdfs = [f'output/{uuid}/report_ref.pdf', f'output/{uuid}/report_batch.pdf']

        # merger = PdfMerger()

        # for pdf in pdfs:
        #     merger.append(pdf)

        # merger.write(f'output/{uuid}/report.pdf')
        # merger.close()
        # options = {'width': 1280, 'disable-smart-width': ''}
        imgkit.from_file(f'output/{uuid}/df_batch.html', f'output/{uuid}/out.jpg', )
        return f'output/{uuid}/out.jpg', df



