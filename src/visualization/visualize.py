import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np

### History Function
def plot_history(history):
       
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    plt.show();
    
## Displaying the results Function
def display_results(y_true, y_preds, class_labels):
    
    results = pd.DataFrame(precision_recall_fscore_support(y_true, y_preds),
                          columns=class_labels).T
    results.rename(columns={0: 'Precision',
                           1: 'Recall',
                           2: 'F-Score',
                           3: 'Support'}, inplace=True)
    
    conf_mat = pd.DataFrame(confusion_matrix(y_true, y_preds), 
                            columns=class_labels,
                            index=class_labels)    
    f2 = fbeta_score(y_true, y_preds, beta=2, average='micro')
    accuracy = accuracy_score(y_true, y_preds)
    print(f"Accuracy: {accuracy}")
    print(f"Global F2 Score: {f2}")    
    return results, conf_mat



def plot_label(mask, labels, col_dict, ax, fig, colorbar = False):
	# Let's also design our color mapping: 1s should be plotted in blue, 2s in red, etc...
	# col_dict={1:"blue",
	#           2:"red",
	#           3:"orange",
	#           4:"green",
	#           5:"yellow",
	#           6:"purple",
	#           7:"grey"}
	# Let's also define the description of each category : 1 (blue) is Sea; 2 (red) is burnt, etc... Order should be respected here ! Or  using another dict maybe could help.
	# labels = np.array(['urban_land','agriculture_land','rangeland','forest_land','water','barren_land','unknown'])
    
	# We create a colormar from our list of colors
	cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
	
	len_lab = len(labels)
	
	# prepare normalizer
	## Prepare bins for the normalizer
	norm_bins = np.sort([*col_dict.keys()]) + 0.5
	norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
# 	print(norm_bins)
	## Make normalizer and formatter
	norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
	fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
	
	# Plot our figure
	im = ax.imshow(mask, cmap=cm, norm=norm, interpolation = 'nearest')
    
	diff = norm_bins[1:] - norm_bins[:-1]
	tickz = norm_bins[:-1] + diff / 2
	if colorbar:
		cb = fig.colorbar(im, format=fmt, ticks=tickz)
	return im, fmt, tickz