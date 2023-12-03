import pandas as pd
from IPython.display import display, HTML
import os
import pdfkit
from pypdf import PdfMerger

def count_instance(result, filenames, uuid, width_list, orientation_list, image_path, reference):
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
        data = {
            'Index': [],
            'FileName': [],
            'Orientation': [],
            'Width': [],
            'Instance': []
        }
        
        df_ref = pd.DataFrame({'Reference': [f'<img src="{ref}" width="640" >' for ref in reference]})


        df = pd.DataFrame(data)

        # Populate the dataframe with counts, width, and orientation
        for i, res in enumerate(result):
            instance_count = len(res)
            df.loc[i] = [i, os.path.basename(filenames[i]), orientation_list[i], width_list[i], instance_count]

        # Reorder columns
        df = df[['Index', 'FileName', 'Orientation', 'Width', 'Instance']]

        # Create a new dataframe (df2) with all columns from df
        df2 = df.copy()

        # Add another column for the image (modify as per your requirement)
        print("IMG PATHS")
        print(image_path)
        base_path = [os.path.basename(path) for path in image_path]
        df2['Image'] = base_path
        # convert your links to html tags 
        def path_to_image_html(path):
            return '<img src="'+ path + '" width="240" >'
        
        pd.set_option('display.max_colwidth', None)

        image_cols = ['Image']

        format_dict = {}
        for image_col in image_cols:
            format_dict[image_col] = path_to_image_html

        
        col_widths = [50, 100, 50, 50, 50, 120] 
        
        # Create the HTML file
        df_html = df2.to_html(f'output/{uuid}/df_batch.html', escape=False, formatters=format_dict, col_space=col_widths)
        df_refs = df_ref.to_html(f'output/{uuid}/df_ref.html', escape=False)
        # Save the modified dataframe to a CSV file
        # df2.to_csv(csv_filename, index=False)
        html_table = HTML(df2.to_html(escape=False))
        display(html_table)
        
        pdfkit.from_file(f'output/{uuid}/df_batch.html', f'output/{uuid}/report_batch.pdf')
        pdfkit.from_file(f'output/{uuid}/df_ref.html', f'output/{uuid}/report_ref.pdf')

        

        pdfs = [f'output/{uuid}/report_ref.pdf', f'output/{uuid}/report_batch.pdf']

        merger = PdfMerger()

        for pdf in pdfs:
            merger.append(pdf)

        merger.write(f'output/{uuid}/report.pdf')
        merger.close()
        return f'output/{uuid}/report.pdf', df



