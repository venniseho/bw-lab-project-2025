import pandas as pd
import matplotlib.pyplot as plt

# Load data from the CSV file into a Pandas DataFrame
df = pd.read_csv('outputs/tests/manual_coco_check/metrics/sam_iou.csv')

# Create the scatter plot
# Specify the columns for the x and y axes
plt.scatter(x=df['iou_orig'], y=df['iou_frag'])

# Add labels and a title for clarity
plt.xlabel('iou on original photo')
plt.ylabel('iou on fragmented photo')
plt.title('iou on original vs fragmented photos')

# x and y axis limits
plt.set_aspect('equal', adjustable='box')
plt.xlim(0, 1)
plt.ylim(0, 1)

# Save the plot
plt.savefig('outputs/tests/manual_coco_check/metrics/sam_iou_scatterplot.png')