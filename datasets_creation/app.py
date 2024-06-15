import gradio as gr
import argparse
import os
from get_dataset import get_dataset
from save_to_list import save_to_list
from save_to_pkl import save_to_pkl
from save_to_grouped_list import save_to_grouped_list
from save_to_mysql import save_to_mysql


base_url = 'https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/'

# Define the directory of HDD_dataset, it is inside project folder, and now parallel with the 'algorithms' and 'datasets_creation' folders
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, '..', 'HDD_dataset')

# Define generated files directory
output_dir = os.path.join(script_dir, '..', 'output')

data_iface = gr.Interface(
    fn=get_dataset,
    inputs=[
        gr.CheckboxGroup(choices=['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'], value=['2013'], label='Years', info='Select the years to consider.'),
        gr.Textbox(label="Base Path", value=base_path, info='Enter the base path to save the data.'),
        gr.Textbox(label="Base URL", value=base_url, info='Enter the base URL to download the data.'),
    ],
    outputs=gr.Textbox(placeholder="See result below.", label="Result"),
    description="Download the data.",
)

list_iface = gr.Interface(
    fn=save_to_list,
    inputs=[
        gr.Textbox(value='ST3000DM001', label='Model', info='Enter the model type(s) for training. For multiple models, separate them with commas.'),
        gr.CheckboxGroup(choices=['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'], value=['2013'], label='Years', info='Select the years to consider.'),
        gr.Checkbox(value=False, label='Generate Only Failed Models', info='Generate only failed models.'),
        gr.Textbox(label="Base Path", value=base_path, info='Enter the base path to save the data.'),
        gr.Textbox(label="Output Directory", value=output_dir, info='Enter the output directory to save the data.'),
    ],
    outputs=gr.Textbox(placeholder="See result below.", label="Result"),
    description="Save the data to a list.",
)

pkl_iface = gr.Interface(
    fn=save_to_pkl,
    inputs=[
        gr.Textbox(value='ST3000DM001', label='Model', info='Enter the model type(s) for training. For multiple models, separate them with commas.'),
        gr.CheckboxGroup(choices=['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'], value=['2013'], label='Years', info='Select the years to consider.'),
        gr.Checkbox(value=False, label='Generate Only Failed Models', info='Generate only failed models.'),
        gr.Textbox(label="Base Path", value=base_path, info='Enter the base path to save the data.'),
        gr.Textbox(label="Output Directory", value=output_dir, info='Enter the output directory to save the data.'),
    ],
    outputs=gr.Textbox(placeholder="See result below.", label="Result"),
    description="Save the data to a pickle file.",
)

mysql_iface = gr.Interface(
    fn=save_to_mysql,
    inputs=[
        gr.Textbox(label='Database User', value='root', info='Enter the database user name.'),
        gr.Textbox(label='Database Password', type='password', value='password', info='Enter the database password.'),
        gr.Textbox(label='Database Host', value='localhost', info='Enter the database host.'),
        gr.Textbox(label='Database Port', value='3306', info='Enter the database port.'),
        gr.Textbox(label='Database Name', value='smart_info', info='Enter the database name.'),
        gr.Textbox(value='ST3000DM001', label='Model', info='Enter the model type(s) for training. For multiple models, separate them with commas.'),
        gr.CheckboxGroup(choices=['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'], value=['2013'], label='Years', info='Select the years to consider.'),
        gr.Checkbox(value=False, label='Generate Only Failed Models', info='Generate only failed models.'),
        gr.Textbox(label="Base Path", value=base_path, info='Enter the base path to save the data.'),
        gr.Textbox(label="Output Directory", value=output_dir, info='Enter the output directory to save the data.'),
    ],
    outputs=gr.Textbox(placeholder="See result below.", label="Result"),
    description="Save the data to a MySQL database.",
)

grouped_list_iface = gr.Interface(
    fn=save_to_grouped_list,
    inputs=[
        gr.Textbox(value='ST3000DM001', label='Model', info='Enter the model type(s) for training. For multiple models, separate them with commas.'),
        gr.CheckboxGroup(choices=['2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'], value=['2013'], label='Years', info='Select the years to consider.'),
        gr.Checkbox(value=False, label='Generate Only Failed Models', info='Generate only failed models.'),
        gr.Textbox(label="Output Directory", value=output_dir, info='Enter the output directory to save the data.'),
    ],
    outputs=gr.Textbox(placeholder="See result below.", label="Result"),
    description="Save the data to a pickle file as list.",
)

demo = gr.TabbedInterface(
    [data_iface, list_iface, pkl_iface, mysql_iface, grouped_list_iface],
    ["Download Data", "Save to List", "Save to PKL", "Save to MySQL Database", "Save to Grouped List"],
    title="Prognostika - Hard Disk Failure Prediction Model Dataset Creation Dashboard",
)

if __name__ == "__main__":
    demo.launch()