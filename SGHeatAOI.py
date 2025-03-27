import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize Tkinter root
root = tk.Tk ()
root.title ("Heatmap and AOI Analysis Tool")
root.geometry ("1300x700")
root.state ('normal')
root.configure (bg='#D3D3D3') # Change root background to lighter grey

# Style configuration
style = ttk.Style ()
style.configure ('TButton', background='lightcoral', foreground='black', padding=3)

# Create frames with lighter grey background
main_frame = tk.Frame (root, bg='#D3D3D3', bd=2, relief=tk.SUNKEN)
main_frame.pack (fill=tk.BOTH, expand=1, padx=3, pady=3)

control_frame = tk.Frame (main_frame, bg='#D3D3D3', bd=2, relief=tk.SUNKEN)
control_frame.pack (side=tk.LEFT, fill=tk.Y, padx=3, pady=3)

image_frame = tk.Frame (main_frame, bg='#D3D3D3', bd=2, relief=tk.SUNKEN)
image_frame.pack (side=tk.RIGHT, fill=tk.BOTH, expand=1, padx=3, pady=3)

# Variables
img_with_polygons = None
img = None
gray = None
original_img = None
contours = []
contour_area_percentage = {}
max_pixel_value = 255
display_img = None
total_grayscale_sum = 0

# Update widgets to have lighter grey background
data_file_entry = tk.Entry (control_frame, width=20, bd=2, relief=tk.GROOVE)
image_file_entry = tk.Entry (control_frame, width=20, bd=2, relief=tk.GROOVE)
spacing_entry = tk.Entry (control_frame, bd=2, relief=tk.GROOVE, width=10)
max_hor_entry = tk.Entry (control_frame, bd=2, relief=tk.GROOVE, width=10)
max_ver_entry = tk.Entry (control_frame, bd=2, relief=tk.GROOVE, width=10)
kernel_size_entry = tk.Entry (control_frame, bd=2, relief=tk.GROOVE, width=10)
s_gaussian_entry = tk.Entry (control_frame, bd=2, relief=tk.GROOVE, width=10)
img_label = tk.Label (image_frame, bg='#D3D3D3', bd=2, relief=tk.SUNKEN) # Updated to lighter grey
threshold_scale = tk.Scale (control_frame, from_=1, to=100, orient=tk.HORIZONTAL, bg='#D3D3D3', bd=2, relief=tk.GROOVE, length=120)
transparency_scale = tk.Scale (control_frame, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.01, bg='#D3D3D3', bd=2, relief=tk.GROOVE, length=120)
show_heatmap_var = tk.IntVar ()

# Progress bar
progress_bar = ttk.Progressbar (control_frame, orient="horizontal", length=200, mode="determinate")

# Threshold label for dynamic display
threshold_label = tk.Label (control_frame, text="Threshold: 1% (0)", bg='#D3D3D3')

# Statistic Variables
min_var = tk.IntVar (value=1)
max_var = tk.IntVar (value=1)
mean_var = tk.IntVar (value=1)
median_var = tk.IntVar (value=1)
value_percentage_var = tk.IntVar (value=1)
area_percentage_var = tk.IntVar (value=1)
ratio_index_var = tk.IntVar (value=1)

# Function to convert OpenCV image to Tkinter-compatible format
def cv2_to_tk (image):
 image = Image.fromarray (image)
 image = ImageTk.PhotoImage (image)
 return image

# Function to filter data that exceeds the image resolution
def filter_data_by_resolution (records, max_hor, max_ver):
 """
 Filters out x, y values in records that exceed the image resolution.
 Also removes any negative values.
 :param records: numpy array with shape (n, 2) containing x, y coordinates.
 :param max_hor: Maximum horizontal resolution (width).
 :param max_ver: Maximum vertical resolution (height).
 :return: Filtered records within image bounds.
 """
 # Filter out any points where the x or y values exceed the image bounds or are negative
 filtered_records = records[ (records[:, 0] >= 0) & (records[:, 1] >= 0) & (records[:, 0] <= max_hor) & (records[:, 1] <= max_ver)]
 return filtered_records

# Function to generate TXT report for the AOIs
def generate_report_txt (data_filename, output_folder):
 report_filename = os.path.join (output_folder, os.path.basename (data_filename).replace ('.txt', '_report.txt'))

 with open (report_filename, 'w') as f:
  # Header
  f.write ("AOI Analysis Report\n")
  f.write (f"Source Data: {data_filename}\n\n")
  
  f.write ("AOI Statistics:\n")
  f.write (f"{'Threshold':<12} {'AOI':<5} {'Min':<10} {'Max':<10} {'Mean':<10} {'Median':<10} {'Value %':<12} {'Area %':<12} {'Ratio Index':<12}\n")
  f.write ("="*90 + "\n")
  
  total_aoi_count = 0

  for threshold in range (1, 101): # Loop over thresholds (you can customize this range)
   update_threshold (threshold)
   for idx, contour in enumerate (contours):
    if len (contour) > 2:
     contour = np.squeeze (contour, axis=1)
     if len (contour) > 3:
      tck, u = splprep ([contour[:, 0], contour[:, 1]], s=0.0)
      x, y = splev (np.linspace (0, 1, 100), tck)
      smoothed_contour = np.column_stack ( (x, y)).astype (np.int32)
     else:
      smoothed_contour = contour
     
     mask = np.zeros_like (gray)
     cv2.drawContours (mask, [smoothed_contour], -1, 255, -1)
     aoi_pixels = gray[mask == 255]
     aoi_sum_values = np.sum (np.where (mask != 0, gray, 0))
     aoi_area = cv2.contourArea (smoothed_contour)
     ratio_index = aoi_sum_values / aoi_area if aoi_area != 0 else 0
     value_percentage = (aoi_sum_values / total_grayscale_sum) * 100
     area_percentage = (aoi_area / (original_img.shape[0] * original_img.shape[1])) * 100
     
     min_value = np.min (aoi_pixels)
     max_value = np.max (aoi_pixels)
     mean_value = np.mean (aoi_pixels)
     median_value = np.median (aoi_pixels)

     # Write the statistics for each AOI to the TXT file
     f.write (f"{threshold:<12} {idx + 1:<5} {min_value:<10.2f} {max_value:<10.2f} {mean_value:<10.2f} {median_value:<10.2f} {value_percentage:<12.2f} {area_percentage:<12.2f} {ratio_index:<12.2f}\n")
     
     total_aoi_count += 1

  # Add total number of AOIs to the end of the file
  f.write ("\n")
  f.write (f"Total AOIs Detected: {total_aoi_count}\n")

 messagebox.showinfo ("Report", f"TXT Report saved to {report_filename}")

# Update image display with heatmap and AOI
def update_image_display ():
 global img_with_polygons, img, original_img, gray, contours, contour_area_percentage, display_img, total_grayscale_sum
 
 display_img = original_img.copy ()

 if show_heatmap_var.get () == 1 and img is not None:
  heatmap_overlay = cv2.addWeighted (display_img, 1 - transparency_scale.get (), img, transparency_scale.get (), 0)
  display_img = heatmap_overlay

 placed_text_positions = []

 if img_with_polygons is not None:
  for contour in contours:
   smoothed_contour = np.array (contour, dtype=np.int32)
   cv2.polylines (display_img, [smoothed_contour], isClosed=True, color= (0, 255, 0), thickness=2)
   M = cv2.moments (smoothed_contour)
   if M["m00"] != 0:
    cX = int (M["m10"] / M["m00"])
    cY = int (M["m01"] / M["m00"])

    mask = np.zeros_like (gray)
    cv2.drawContours (mask, [smoothed_contour], -1, 255, -1)
    aoi_pixels = gray[mask == 255]

    min_value = np.min (aoi_pixels)
    max_value = np.max (aoi_pixels)
    mean_value = np.mean (aoi_pixels)
    median_value = np.median (aoi_pixels)
    aoi_sum_values = np.sum (aoi_pixels)
    aoi_area = cv2.contourArea (smoothed_contour)

    ratio_index = aoi_sum_values / aoi_area if aoi_area != 0 else 0
    value_percentage = (aoi_sum_values / total_grayscale_sum) * 100
    area_percentage = (aoi_area / (display_img.shape[0] * display_img.shape[1])) * 100

    text_lines = []
    if min_var.get ():
     text_lines.append (f"Min: {min_value}")
    if max_var.get ():
     text_lines.append (f"Max: {max_value}")
    if mean_var.get ():
     text_lines.append (f"Mean: {mean_value:.2f}")
    if median_var.get ():
     text_lines.append (f"Median: {median_value}")
    if value_percentage_var.get ():
     text_lines.append (f"Value %: {value_percentage:.2f}")
    if area_percentage_var.get ():
     text_lines.append (f"Area %: {area_percentage:.2f}")
    if ratio_index_var.get ():
     text_lines.append (f"Ratio Index: {ratio_index:.2f}")

    text_height = 20
    total_text_height = len (text_lines) * text_height
    max_text_width = 200

    if cX + max_text_width > display_img.shape[1]:
     cX -= (max_text_width + 10)
    if cY + total_text_height > display_img.shape[0]:
     cY -= (total_text_height + 10)
    if cX < 0:
     cX = 10
    if cY < 0:
     cY = 10

    # New: Limit to avoid infinite loops
    max_attempts = 5 # Define a limit for how many times to attempt placement
    attempts = 0
    overlap_found = True
    while overlap_found and attempts < max_attempts:
     overlap_found = False
     for (px, py, pheight, pwidth) in placed_text_positions:
      if (abs (cX - px) < pwidth) and (abs (cY - py) < pheight):
       cY = py + pheight + 10
       if cY + total_text_height > display_img.shape[0]:
        cY = py - total_text_height - 10
       overlap_found = True
       break
     attempts += 1

    placed_text_positions.append ( (cX, cY, total_text_height, max_text_width))

    # Now place the text
    for i, line in enumerate (text_lines):
     color = (255, 255, 255)
     if "Min" in line:
      color = (255, 0, 0)
     elif "Max" in line:
      color = (0, 255, 0)
     elif "Mean" in line:
      color = (0, 0, 255)
     elif "Median" in line:
      color = (255, 255, 0)
     elif "Value %" in line:
      color = (255, 0, 255)
     elif "Area %" in line:
      color = (0, 255, 255)
     elif "Ratio Index" in line:
      color = (128, 0, 128)

     cv2.putText (display_img, line, (cX, cY + i * text_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)

 display_img = cv2.cvtColor (display_img, cv2.COLOR_BGR2RGB)
 img_tk = cv2_to_tk (display_img)
 img_label.config (image=img_tk)
 img_label.image = img_tk

# Function to update threshold
def update_threshold (value):
 global img_with_polygons, gray, original_img, contours, contour_area_percentage, total_heatmap_sum

 if gray is None:
  messagebox.showerror ("Error", "No heatmap available. Please generate a heatmap first.")
  return

 threshold_value = int (int (value) * 255 / 100)
 threshold_label.config (text=f"Threshold: {value}% ({threshold_value})")
 _, thresholded = cv2.threshold (gray, threshold_value, 255, cv2.THRESH_BINARY)
 contours, _ = cv2.findContours (thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
 total_pixel_values = np.sum (gray)
 total_heatmap_sum = np.sum (gray)
 contour_area_percentage = {}

 for contour in contours:
  if len (contour) > 2:
   contour = np.squeeze (contour, axis=1)
   if len (contour) > 3:
    tck, u = splprep ([contour[:, 0], contour[:, 1]], s=0.0)
    x, y = splev (np.linspace (0, 1, 100), tck)
    smoothed_contour = np.column_stack ( (x, y)).astype (np.int32)
   else:
    smoothed_contour = contour
   
   mask = np.zeros_like (gray)
   cv2.drawContours (mask, [smoothed_contour], -1, 255, -1)
   aoi_pixel_values = np.sum (np.where (mask != 0, gray, 0))
   
   percentage = (aoi_pixel_values / total_heatmap_sum) * 100
   contour_area_percentage[tuple (contour.ravel ())] = percentage

 update_image_display ()

# Helper function to count points in a region
def points_in_region (x, y, left_up_edge_x, left_up_edge_y, right_down_edge_x, right_down_edge_y):
 return np.sum ( (x >= left_up_edge_x) & (x <= right_down_edge_x) & (y >= left_up_edge_y) & (y <= right_down_edge_y))

# Update image dimension defaults
def update_image_defaults (image_path):
 global max_hor_entry, max_ver_entry
 img = cv2.imread (image_path)
 if img is not None:
  height, width = img.shape[:2]
  max_hor_entry.delete (0, tk.END)
  max_hor_entry.insert (0, str (width))
  max_ver_entry.delete (0, tk.END)
  max_ver_entry.insert (0, str (height))

# File browser helper
def browse_file (entry, filetypes):
 filename = filedialog.askopenfilename (filetypes=filetypes)
 entry.delete (0, tk.END)
 entry.insert (0, filename)

# Generate heatmap and save images with AOIs and stats
def generate_heatmap (data_filename, image_filename, output_folder):
 global img, gray, img_with_polygons, original_img, max_pixel_value, contours, contour_area_percentage, total_heatmap_sum, total_grayscale_sum

 try:
  records = np.loadtxt (data_filename)
 except IOError:
  messagebox.showerror ("Error", f"File {data_filename} not accessible.")
  return
 except ValueError:
  messagebox.showerror ("Error", f"File {data_filename} has invalid format.")
  return

 progress_bar['value'] = 0
 root.update_idletasks ()

 # Extract X, Y coordinates
 max_hor = int (max_hor_entry.get ())
 max_ver = int (max_ver_entry.get ())
 
 filtered_records = filter_data_by_resolution (records, max_hor, max_ver)

 if len (filtered_records) == 0:
  messagebox.showerror ("Error", "No valid data points within image bounds.")
  return

 x = filtered_records[:, 0]
 y = filtered_records[:, 1]

 spacing = int (spacing_entry.get ())
 kernel_size_gaussian = int (kernel_size_entry.get ())
 s_gaussian = int (s_gaussian_entry.get ())

 # Prepare the grid for the heatmap
 heatmap_ver_values = np.arange (0, max_ver + spacing, spacing)
 heatmap_hor_values = np.arange (0, max_hor + spacing, spacing)

 heatmap_ver_number = len (heatmap_ver_values) - 1
 heatmap_hor_number = len (heatmap_hor_values) - 1

 # Create a zero matrix for the heatmap
 heatmap = np.zeros ( (heatmap_ver_number, heatmap_hor_number))

 progress_bar['maximum'] = heatmap_ver_number
 root.update_idletasks ()

 # Calculate points in each region for the heatmap
 for l in range (heatmap_ver_number):
  for k in range (heatmap_hor_number):
   heatmap[l, k] = points_in_region (
    x, y, 
    heatmap_hor_values[k], heatmap_ver_values[l], 
    heatmap_hor_values[k + 1], heatmap_ver_values[l + 1]
   )
  progress_bar['value'] += 1
  root.update_idletasks ()

 # Apply Gaussian filter
 heatmap = gaussian_filter (
  heatmap, 
  sigma=s_gaussian, 
  mode='nearest', 
  truncate= (kernel_size_gaussian / (2 * s_gaussian))
 )

 # Normalize the heatmap to fit into an 8-bit image range (0-255)
 max_pixel_value = np.max (heatmap)
 heatmap = 255 * heatmap / max_pixel_value
 heatmap = np.floor (heatmap).astype (np.uint8)

 # Save heatmap image
 heatmap_output_path = os.path.join (output_folder, os.path.basename (data_filename).replace ('.txt', '_heatmap.png'))
 cv2.imwrite (heatmap_output_path, heatmap)
 img = cv2.imread (heatmap_output_path)
 gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
 total_grayscale_sum = np.sum (gray)
 original_img = cv2.imread (image_filename)

 if original_img is None:
  messagebox.showerror ("Error", f"File {image_filename} not accessible.")
  return

 img_with_polygons = original_img.copy ()
 contours = []
 contour_area_percentage = {}

 thresholds = [20, 40, 60, 80]
 
 # Loop for images with heatmap as background
 for threshold in thresholds:
  update_threshold (threshold)

  # Ensure heatmap and AOIs are combined
  show_heatmap_var.set (1)
  update_image_display ()

  # Save image with both AOIs and heatmap
  display_img_bgr = cv2.cvtColor (display_img, cv2.COLOR_RGB2BGR)
  combined_output_path = os.path.join (output_folder, os.path.basename (data_filename).replace ('.txt', f'_combined_{threshold}.png'))
  cv2.imwrite (combined_output_path, display_img_bgr)

 # Loop for images with original image as background
 for threshold in thresholds:
  update_threshold (threshold)

  # Ensure that only the AOIs and statistics are shown (no heatmap)
  show_heatmap_var.set (0)
  update_image_display ()

  # Save image with AOIs and statistics over original image
  display_img_bgr = cv2.cvtColor (display_img, cv2.COLOR_RGB2BGR)
  original_output_path = os.path.join (output_folder, os.path.basename (data_filename).replace ('.txt', f'_original_{threshold}.png'))
  cv2.imwrite (original_output_path, display_img_bgr)

 # Generate the TXT report
 generate_report_txt (data_filename, output_folder)

 # Progress bar completion
 progress_bar['value'] = progress_bar['maximum']
 root.update_idletasks ()

# Batch processing of data and image files
def batch_process ():
 input_folder = filedialog.askdirectory (title="Select Input Folder")
 output_folder = filedialog.askdirectory (title="Select Output Folder")

 if not input_folder or not output_folder:
  messagebox.showerror ("Error", "Input or output folder not selected.")
  return

 data_files = sorted ([os.path.join (input_folder, f) for f in os.listdir (input_folder) if f.endswith ('.txt')])
 image_files = sorted ([os.path.join (input_folder, f) for f in os.listdir (input_folder) if f.endswith ( ('.png', '.jpg', '.jpeg'))])

 if len (data_files) != len (image_files):
  messagebox.showerror ("Error", "The number of data files and image files must be the same.")
  return

 progress_bar['value'] = 0
 progress_bar['maximum'] = len (data_files)
 root.update_idletasks ()

 for data_file, image_file in zip (data_files, image_files):
  generate_heatmap (data_file, image_file, output_folder)
  progress_bar['value'] += 1
  root.update_idletasks ()

 progress_bar['value'] = progress_bar['maximum']
 root.update_idletasks ()

# GUI elements
tk.Label (control_frame, text="Data File:", bg='#D3D3D3').grid (row=0, column=0, padx=3, pady=3, sticky='w')
data_file_entry.grid (row=0, column=1, padx=3, pady=3)
ttk.Button (control_frame, text="Browse", command=lambda: browse_file (data_file_entry, [ ("Text files", "*.txt"), ("All files", "*.*")])).grid (row=0, column=2, padx=3, pady=3)

tk.Label (control_frame, text="Image File:", bg='#D3D3D3').grid (row=1, column=0, padx=3, pady=3, sticky='w')
image_file_entry.grid (row=1, column=1, padx=3, pady=3)
ttk.Button (control_frame, text="Browse", command=lambda: [browse_file (image_file_entry, [ ("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")]), update_image_defaults (image_file_entry.get ())]).grid (row=1, column=2, padx=3, pady=3)

ttk.Button (control_frame, text="Batch Process", command=batch_process).grid (row=2, column=0, columnspan=3, padx=3, pady=3)

batch_message = tk.Label (control_frame, text="For batch process put all data in one folder (.txt, .jpg)", bg='#D3D3D3')
batch_message.grid (row=3, column=0, columnspan=3, padx=3, pady=3)

tk.Label (control_frame, text="Spacing (pixels):", bg='#D3D3D3').grid (row=4, column=0, padx=3, pady=3, sticky='w')
spacing_entry.grid (row=4, column=1, padx=3, pady=3)
spacing_entry.insert (0, "1")

tk.Label (control_frame, text="Max Hor (pixels):", bg='#D3D3D3').grid (row=5, column=0, padx=3, pady=3, sticky='w')
max_hor_entry.grid (row=5, column=1, padx=3, pady=3)
max_hor_entry.insert (0, "1280")

tk.Label (control_frame, text="Max Ver (pixels):", bg='#D3D3D3').grid (row=6, column=0, padx=3, pady=3, sticky='w')
max_ver_entry.grid (row=6, column=1, padx=3, pady=3)
max_ver_entry.insert (0, "1024")

tk.Label (control_frame, text="Kernel Size:", bg='#D3D3D3').grid (row=7, column=0, padx=3, pady=3, sticky='w')
kernel_size_entry.grid (row=7, column=1, padx=3, pady=3)
kernel_size_entry.insert (0, "121")

tk.Label (control_frame, text="Sigma:", bg='#D3D3D3').grid (row=8, column=0, padx=3, pady=3, sticky='w')
s_gaussian_entry.grid (row=8, column=1, padx=3, pady=3)
s_gaussian_entry.insert (0, "30")

ttk.Button (control_frame, text="Generate Heatmap", command=lambda: generate_heatmap (data_file_entry.get (), image_file_entry.get (), filedialog.askdirectory (title="Select Output Folder"))).grid (row=9, column=0, columnspan=3, padx=3, pady=3)

title_label = tk.Label (image_frame, text="Image Preview for the selected Threshold", bg='#D3D3D3', fg='black', font= ("Helvetica", 12, "bold"))
title_label.pack (padx=3, pady=3)

img_label.pack (padx=3, pady=3, fill=tk.BOTH, expand=1)

threshold_label.grid (row=10, column=0, padx=3, pady=3, sticky='w')
threshold_scale.grid (row=10, column=1, padx=3, pady=3)
threshold_scale.config (command=lambda value: update_threshold (value))

tk.Label (control_frame, text="Transparency:", bg='#D3D3D3').grid (row=11, column=0, padx=3, pady=3, sticky='w')
transparency_scale.grid (row=11, column=1, padx=3, pady=3)
transparency_scale.set (1)
transparency_scale.config (command=lambda value: update_image_display ())

tk.Checkbutton (control_frame, text="Show Heatmap", variable=show_heatmap_var, command=update_image_display, bg='#D3D3D3').grid (row=12, column=0, padx=3, pady=3)

ttk.Button (control_frame, text="Save Current Preview As", command=lambda: save_current_preview_as ()).grid (row=13, column=0, columnspan=2, padx=3, pady=3)
ttk.Button (control_frame, text="Save Heatmap", command=lambda: save_heatmap ()).grid (row=14, column=0, columnspan=2, padx=3, pady=3)

ttk.Button (control_frame, text="Report Export", command=lambda: generate_report_txt (data_file_entry.get (), filedialog.askdirectory (title="Select Output Folder"))).grid (row=15, column=0, columnspan=2, padx=3, pady=3)

tk.Label (control_frame, text="Select Statistics:", bg='#D3D3D3').grid (row=18, column=0, padx=3, pady=3, sticky='w')
tk.Checkbutton (control_frame, text="Min", variable=min_var, command=update_image_display, bg='#D3D3D3').grid (row=19, column=0, padx=3, pady=3)
tk.Checkbutton (control_frame, text="Max", variable=max_var, command=update_image_display, bg='#D3D3D3').grid (row=19, column=1, padx=3, pady=3)
tk.Checkbutton (control_frame, text="Mean", variable=mean_var, command=update_image_display, bg='#D3D3D3').grid (row=19, column=2, padx=3, pady=3)
tk.Checkbutton (control_frame, text="Median", variable=median_var, command=update_image_display, bg='#D3D3D3').grid (row=19, column=3, padx=3, pady=3)
tk.Checkbutton (control_frame, text="Value %", variable=value_percentage_var, command=update_image_display, bg='#D3D3D3').grid (row=20, column=0, padx=3, pady=3)
tk.Checkbutton (control_frame, text="Area %", variable=area_percentage_var, command=update_image_display, bg='#D3D3D3').grid (row=20, column=1, padx=3, pady=3)
tk.Checkbutton (control_frame, text="Ratio Index", variable=ratio_index_var, command=update_image_display, bg='#D3D3D3').grid (row=20, column=2, padx=3, pady=3)

progress_bar.grid (row=16, column=0, columnspan=3, padx=3, pady=3)

copyright_label = tk.Label (control_frame, text="© Koutras Efthymis 2024", bg='#D3D3D3')
copyright_label.grid (row=17, column=0, columnspan=3, sticky="se", padx=3, pady=3)

root.mainloop ()
