import os
import pandas as pd
import scipy.io as io

path = os.path.join(os.getcwd(), 'Annotations')

folders = sorted(os.listdir(path))
column=['Image','x_top','y_top','x_bottom','y_bottom','label']

for dir in folders:
    data = pd.DataFrame(columns = column)
    folder = os.path.join(path, dir)

    for file in sorted(os.listdir(folder)):
        mat = io.loadmat(os.path.join(folder, file))
        df = pd.DataFrame(['image_'+file[11:15]+'.jpg', mat['box_coord'][0][0], mat['box_coord'][0][1], mat['box_coord'][0][2], mat['box_coord'][0][3], dir]).T
        df.columns = column
        data = data.append(df)

    data.to_csv(path_or_buf= os.path.join(os.getcwd(), 'dataset', 'annotations_csv') + '//' + dir + '.csv')
