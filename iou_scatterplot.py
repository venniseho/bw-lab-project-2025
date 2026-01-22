import pandas as pd
import matplotlib.pyplot as plt

# 1. Load data from the CSV file into a Pandas DataFrame
df = pd.read_csv('outputs/tests/manual_coco_check/metrics/sam_iou.csv')

# 2. Create the scatter plot
# Specify the columns for the x and y axes
plt.scatter(x=df['iou_orig'], y=df['iou_frag'])

# 3. Add labels and a title for clarity
plt.xlabel('iou on original photo')
plt.ylabel('iou on fragmented photo')
plt.title('iou on original vs fragmented photos')

# 4. save the plot
plt.savefig('outputs/tests/manual_coco_check/metrics/sam_iou_scatterplot.png')